"""SVO-enhanced PPO implementation for SocialJax environments."""
from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

import socialjax
from socialjax.wrappers.baselines import LogWrapper

from components.algorithms.networks import ActorCritic, EncoderConfig
from components.shaping.svo import svo_deviation_penalty, svo_linear_combination
from components.training.logging import finalize_info_stats, init_wandb, update_info_stats
from components.training.ppo import PPOBatch, compute_gae, update_ppo
from components.training.utils import (
    flatten_obs,
    to_actor_dones,
    to_actor_rewards,
    unflatten_actions,
)


def _done_dict_to_array(done: Dict, agents: List[int]) -> jnp.ndarray:
    return jnp.stack([done[str(a)] for a in agents], axis=1)


def _normalize_svo_param(value, num_agents, name):
    if isinstance(value, (list, tuple, jnp.ndarray)):
        arr = jnp.asarray(value, dtype=jnp.float32)
        if arr.shape != (num_agents,):
            raise ValueError(f"{name} must have shape ({num_agents},), got {arr.shape}")
        return arr
    return jnp.full((num_agents,), float(value), dtype=jnp.float32)


def _build_target_mask(target_agents, num_agents):
    if target_agents is None:
        return None
    mask = jnp.zeros((num_agents,), dtype=bool)
    return mask.at[jnp.array(target_agents)].set(True)


def _build_encoder_cfg(config: Dict) -> EncoderConfig:
    return EncoderConfig(
        activation=config.get("ACTIVATION", "relu"),
        mlp_sizes=tuple(config.get("MLP_HIDDEN_SIZES", (64, 64))),
        cnn_channels=tuple(config.get("CNN_CHANNELS", (32, 32, 32))),
        cnn_kernel_sizes=tuple(config.get("CNN_KERNEL_SIZES", ((5, 5), (3, 3), (3, 3)))),
        cnn_dense_size=int(config.get("CNN_DENSE_SIZE", 64)),
    )


def make_train(config: Dict):
    env = socialjax.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))
    env = LogWrapper(env, replace_info=False)

    num_envs = int(config["NUM_ENVS"])
    num_steps = int(config["NUM_STEPS"])
    num_agents = env.num_agents
    parameter_sharing = bool(config.get("PARAMETER_SHARING", True))

    if parameter_sharing:
        num_actors = num_envs * num_agents
    else:
        num_actors = num_envs

    num_updates = (
        int(config["TOTAL_TIMESTEPS"]) // num_steps // num_envs
    )
    minibatch_size = (
        num_actors * num_steps // int(config["NUM_MINIBATCHES"])
    )

    svo_mode = config.get("SVO_MODE", "deviation")
    svo_w = _normalize_svo_param(config.get("SVO_W", 0.5), num_agents, "SVO_W")
    svo_ideal = _normalize_svo_param(
        config.get("SVO_IDEAL_ANGLE_DEGREES", 45), num_agents, "SVO_IDEAL_ANGLE_DEGREES"
    )
    svo_angles = _normalize_svo_param(
        config.get("SVO_ANGLE_DEGREES", 45), num_agents, "SVO_ANGLE_DEGREES"
    )
    svo_target_mask = _build_target_mask(config.get("SVO_TARGET_AGENTS"), num_agents)

    encoder_cfg = _build_encoder_cfg(config)

    def _shape_reward(reward: jnp.ndarray):
        if svo_mode == "linear":
            shaped, theta = svo_linear_combination(reward, svo_angles, svo_target_mask)
        else:
            shaped, theta = svo_deviation_penalty(reward, svo_ideal, svo_w, svo_target_mask)
        return shaped, theta

    def train(rng):
        wandb = init_wandb(config)
        log_enabled = wandb is not None

        if not parameter_sharing:
            # Fallback path for independent policies.
            network = [ActorCritic(env.action_space().n, encoder_cfg) for _ in range(num_agents)]
            rng, init_rng = jax.random.split(rng)
            init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
            params = [net.init(init_rng, init_x) for net in network]

            if config.get("ANNEAL_LR", False):
                def lr_schedule(count):
                    frac = (
                        1.0
                        - (count // (int(config["NUM_MINIBATCHES"]) * int(config["UPDATE_EPOCHS"])))
                        / num_updates
                    )
                    return float(config["LR"]) * frac

                tx = optax.chain(
                    optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
                    optax.adam(learning_rate=lr_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
                    optax.adam(float(config["LR"]), eps=1e-5),
                )

            train_state = [
                TrainState.create(
                    apply_fn=network[i].apply,
                    params=params[i],
                    tx=tx,
                )
                for i in range(num_agents)
            ]

            rng, reset_rng = jax.random.split(rng)
            reset_keys = jax.random.split(reset_rng, num_envs)
            obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)

            for update_step in range(num_updates):
                obs_buf = []
                actions_buf = []
                logp_buf = []
                values_buf = []
                rewards_buf = []
                extrinsic_rewards_buf = []
                dones_buf = []
                theta_buf = []
                info_stats: Dict[str, Dict[str, float]] = {}

                for _ in range(num_steps):
                    obs_batch = [obs[:, i] for i in range(num_agents)]
                    env_actions = []
                    action = []
                    logp = []
                    value = []
                    for i in range(num_agents):
                        rng, action_rng = jax.random.split(rng)
                        pi_i, value_i = network[i].apply(train_state[i].params, obs_batch[i])
                        action_i = pi_i.sample(seed=action_rng)
                        env_actions.append(action_i)
                        action.append(action_i)
                        logp.append(pi_i.log_prob(action_i))
                        value.append(value_i)
                    action = jnp.stack(action, axis=0)
                    logp = jnp.stack(logp, axis=0)
                    value = jnp.stack(value, axis=0)

                    rng, step_rng = jax.random.split(rng)
                    step_keys = jax.random.split(step_rng, num_envs)
                    obs, env_state, reward, done, info = jax.vmap(
                        env.step, in_axes=(0, 0, 0)
                    )(step_keys, env_state, env_actions)

                    done_array = _done_dict_to_array(done, env.agents)
                    shaped_reward, theta = _shape_reward(reward)
                    if log_enabled:
                        update_info_stats(info_stats, info)
                        theta_buf.append(theta)

                    obs_buf.append(jnp.stack(obs_batch, axis=0))
                    actions_buf.append(action)
                    logp_buf.append(logp)
                    values_buf.append(value)
                    rewards_buf.append(shaped_reward.T)
                    extrinsic_rewards_buf.append(reward.T)
                    dones_buf.append(done_array.T)

                obs_arr = jnp.stack(obs_buf)
                actions_arr = jnp.stack(actions_buf)
                logp_arr = jnp.stack(logp_buf)
                values_arr = jnp.stack(values_buf)
                rewards_arr = jnp.stack(rewards_buf)
                dones_arr = jnp.stack(dones_buf)

                for agent_idx in range(num_agents):
                    agent_obs = obs_arr[:, agent_idx]
                    agent_actions = actions_arr[:, agent_idx]
                    agent_logp = logp_arr[:, agent_idx]
                    agent_values = values_arr[:, agent_idx]
                    agent_rewards = rewards_arr[:, agent_idx]
                    agent_dones = dones_arr[:, agent_idx]

                    last_obs = obs[:, agent_idx]
                    _, last_value = network[agent_idx].apply(
                        train_state[agent_idx].params, last_obs
                    )

                    advantages, returns = compute_gae(
                        agent_rewards,
                        agent_values,
                        agent_dones,
                        last_value,
                        float(config["GAMMA"]),
                        float(config["GAE_LAMBDA"]),
                    )

                    batch = PPOBatch(
                        obs=agent_obs.reshape((-1, *agent_obs.shape[2:])),
                        actions=agent_actions.reshape((-1,)),
                        old_log_probs=agent_logp.reshape((-1,)),
                        advantages=advantages.reshape((-1,)),
                        returns=returns.reshape((-1,)),
                    )

                    rng, shuffle_rng = jax.random.split(rng)
                    indices = jax.random.permutation(shuffle_rng, batch.actions.shape[0])

                    for _ in range(int(config["UPDATE_EPOCHS"])):
                        for start in range(0, indices.shape[0], minibatch_size):
                            mb_idx = indices[start:start + minibatch_size]
                            mbatch = PPOBatch(
                                obs=batch.obs[mb_idx],
                                actions=batch.actions[mb_idx],
                                old_log_probs=batch.old_log_probs[mb_idx],
                                advantages=batch.advantages[mb_idx],
                                returns=batch.returns[mb_idx],
                            )
                            new_state, _ = update_ppo(
                                train_state[agent_idx],
                                mbatch,
                                float(config["CLIP_EPS"]),
                                float(config["ENT_COEF"]),
                                float(config["VF_COEF"]),
                            )
                            train_state[agent_idx] = new_state

                if log_enabled:
                    metrics = finalize_info_stats(info_stats)
                    metrics["train/svo_reward_mean"] = float(jnp.mean(jnp.stack(rewards_buf)))
                    metrics["train/extrinsic_reward_mean"] = float(jnp.mean(jnp.stack(extrinsic_rewards_buf)))
                    if theta_buf:
                        metrics["svo/theta_mean"] = float(jnp.mean(jnp.stack(theta_buf)))
                    metrics["update_step"] = update_step + 1
                    metrics["env_step"] = (update_step + 1) * num_steps * num_envs
                    wandb.log(metrics, step=metrics["env_step"])

            return train_state

        network = ActorCritic(env.action_space().n, encoder_cfg)
        rng, init_rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
        params = network.init(init_rng, init_x)

        if config.get("ANNEAL_LR", False):
            def lr_schedule(count):
                frac = (
                    1.0
                    - (count // (int(config["NUM_MINIBATCHES"]) * int(config["UPDATE_EPOCHS"])))
                    / num_updates
                )
                return float(config["LR"]) * frac

            tx = optax.chain(
                optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
                optax.adam(learning_rate=lr_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
                optax.adam(float(config["LR"]), eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )

        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)

        def _log_callback(metrics):
            if log_enabled:
                wandb.log(metrics, step=int(metrics["env_step"]))

        def _env_step(carry, _):
            train_state, env_state, last_obs, rng = carry
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            obs_batch = flatten_obs(last_obs)
            pi, value = network.apply(train_state.params, obs_batch)
            action = pi.sample(seed=action_rng)
            logp = pi.log_prob(action)
            env_actions = unflatten_actions(action, num_envs, num_agents)

            step_keys = jax.random.split(step_rng, num_envs)
            obs, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_keys, env_state, env_actions)

            done_array = _done_dict_to_array(done, env.agents)
            shaped_reward, theta = _shape_reward(reward)
            info_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x), info)

            transition = (
                obs_batch,
                action,
                logp,
                value,
                to_actor_rewards(shaped_reward),
                to_actor_rewards(reward),
                to_actor_dones(done_array),
                theta,
                info_mean,
            )
            return (train_state, env_state, obs, rng), transition

        def _update_step(carry, _):
            train_state, env_state, last_obs, rng, update_step = carry
            (train_state, env_state, last_obs, rng), traj = jax.lax.scan(
                _env_step,
                (train_state, env_state, last_obs, rng),
                None,
                length=num_steps,
            )
            (
                obs_arr,
                actions_arr,
                logp_arr,
                values_arr,
                rewards_arr,
                extrinsic_arr,
                dones_arr,
                thetas_arr,
                info_means,
            ) = traj

            last_obs_batch = flatten_obs(last_obs)
            _, last_value = network.apply(train_state.params, last_obs_batch)

            advantages, returns = compute_gae(
                rewards_arr,
                values_arr,
                dones_arr,
                last_value,
                float(config["GAMMA"]),
                float(config["GAE_LAMBDA"]),
            )

            batch = PPOBatch(
                obs=obs_arr.reshape((-1, *obs_arr.shape[2:])),
                actions=actions_arr.reshape((-1,)),
                old_log_probs=logp_arr.reshape((-1,)),
                advantages=advantages.reshape((-1,)),
                returns=returns.reshape((-1,)),
            )

            batch_size = batch.actions.shape[0]

            def _update_epoch(carry, _):
                train_state, rng = carry
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, batch_size)

                def _minibatch(state, idx):
                    start = idx * minibatch_size
                    mb_idx = perm[start:start + minibatch_size]
                    mbatch = PPOBatch(
                        obs=batch.obs[mb_idx],
                        actions=batch.actions[mb_idx],
                        old_log_probs=batch.old_log_probs[mb_idx],
                        advantages=batch.advantages[mb_idx],
                        returns=batch.returns[mb_idx],
                    )
                    state, _ = update_ppo(
                        state,
                        mbatch,
                        float(config["CLIP_EPS"]),
                        float(config["ENT_COEF"]),
                        float(config["VF_COEF"]),
                    )
                    return state, None

                train_state, _ = jax.lax.scan(
                    _minibatch,
                    train_state,
                    jnp.arange(int(config["NUM_MINIBATCHES"])),
                )
                return (train_state, rng), None

            (train_state, rng), _ = jax.lax.scan(
                _update_epoch,
                (train_state, rng),
                None,
                length=int(config["UPDATE_EPOCHS"]),
            )

            info_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), info_means)
            metrics = {
                "train/svo_reward_mean": jnp.mean(rewards_arr),
                "train/extrinsic_reward_mean": jnp.mean(extrinsic_arr),
                "svo/theta_mean": jnp.mean(thetas_arr),
                "update_step": update_step + 1,
                "env_step": (update_step + 1) * num_steps * num_envs,
            }
            for key in info_mean:
                metrics[f"env/{key}"] = info_mean[key]
            jax.debug.callback(_log_callback, metrics)

            return (train_state, env_state, last_obs, rng, update_step + 1), metrics

        init_carry = (train_state, env_state, obs, rng, 0)

        def _train_shared(carry):
            return jax.lax.scan(_update_step, carry, None, length=num_updates)

        final_carry, _ = jax.jit(_train_shared)(init_carry)
        return final_carry[0]

    return train
