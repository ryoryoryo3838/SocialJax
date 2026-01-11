"""MAPPO implementation for SocialJax environments."""
from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

import socialjax
from socialjax.wrappers.baselines import LogWrapper, MAPPOWorldStateWrapper

from components.algorithms.networks import Actor, Critic, EncoderConfig
from components.training.logging import finalize_info_stats, init_wandb, update_info_stats
from components.training.ppo import PPOBatch, compute_gae, update_actor, update_value
from components.training.utils import (
    build_world_state,
    flatten_obs,
    to_actor_dones,
    to_actor_rewards,
    unflatten_actions,
)


def _done_dict_to_array(done: Dict, agents: List[int]) -> jnp.ndarray:
    return jnp.stack([done[str(a)] for a in agents], axis=1)


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
    env = MAPPOWorldStateWrapper(env)
    env = LogWrapper(env, replace_info=False)

    num_envs = int(config["NUM_ENVS"])
    num_steps = int(config["NUM_STEPS"])
    num_agents = env.num_agents
    parameter_sharing = bool(config.get("PARAMETER_SHARING", True))

    num_actors = num_envs * num_agents
    num_updates = (
        int(config["TOTAL_TIMESTEPS"]) // num_steps // num_envs
    )
    minibatch_size = (
        num_actors * num_steps // int(config["NUM_MINIBATCHES"])
    )

    encoder_cfg = _build_encoder_cfg(config)

    def train(rng):
        wandb = init_wandb(config)
        log_enabled = wandb is not None

        actor_net = Actor(env.action_space().n, encoder_cfg)
        critic_net = Critic(encoder_cfg)

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        init_obs = jnp.zeros((1, *(env.observation_space()[0]).shape))
        actor_params = actor_net.init(actor_rng, init_obs)

        world_shape = build_world_state(jnp.zeros((1, num_agents, *env.observation_space()[0].shape))).shape[1:]
        critic_init = jnp.zeros((1, *world_shape))
        critic_params = critic_net.init(critic_rng, critic_init)

        if config.get("ANNEAL_LR", False):
            def lr_schedule(count):
                frac = (
                    1.0
                    - (count // (int(config["NUM_MINIBATCHES"]) * int(config["UPDATE_EPOCHS"])))
                    / num_updates
                )
                return float(config["LR"]) * frac

            actor_tx = optax.chain(
                optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
                optax.adam(learning_rate=lr_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
                optax.adam(learning_rate=lr_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
                optax.adam(float(config["LR"]), eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
                optax.adam(float(config["LR"]), eps=1e-5),
            )

        if parameter_sharing:
            actor_state = TrainState.create(
                apply_fn=actor_net.apply,
                params=actor_params,
                tx=actor_tx,
            )
        else:
            actor_state = [
                TrainState.create(
                    apply_fn=actor_net.apply,
                    params=actor_params,
                    tx=actor_tx,
                )
                for _ in range(num_agents)
            ]

        critic_state = TrainState.create(
            apply_fn=critic_net.apply,
            params=critic_params,
            tx=critic_tx,
        )

        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)

        if not parameter_sharing:
            # Keep independent policy path in Python for now.
            for update_step in range(num_updates):
                obs_buf = []
                actions_buf = []
                logp_buf = []
                values_buf = []
                rewards_buf = []
                dones_buf = []
                world_buf = []
                info_stats: Dict[str, Dict[str, float]] = {}

                for _ in range(num_steps):
                    world_state = build_world_state(obs)
                    world_repeat = jnp.repeat(world_state[:, None, ...], num_agents, axis=1)
                    world_flat = flatten_obs(world_repeat)

                    obs_batch = [obs[:, i] for i in range(num_agents)]
                    env_actions = []
                    action = []
                    logp = []
                    for i in range(num_agents):
                        rng, action_rng = jax.random.split(rng)
                        dist = actor_state[i].apply_fn(actor_state[i].params, obs_batch[i])
                        action_i = dist.sample(seed=action_rng)
                        env_actions.append(action_i)
                        action.append(action_i)
                        logp.append(dist.log_prob(action_i))
                    action = jnp.stack(action, axis=0)
                    logp = jnp.stack(logp, axis=0)

                    value = critic_state.apply_fn(critic_state.params, world_flat)

                    rng, step_rng = jax.random.split(rng)
                    step_keys = jax.random.split(step_rng, num_envs)
                    obs, env_state, reward, done, info = jax.vmap(
                        env.step, in_axes=(0, 0, 0)
                    )(step_keys, env_state, env_actions)

                    done_array = _done_dict_to_array(done, env.agents)
                    value_agents = value.reshape((num_agents, num_envs)).transpose((1, 0))
                    if log_enabled:
                        update_info_stats(info_stats, info)

                    obs_buf.append(jnp.stack(obs_batch, axis=0))
                    actions_buf.append(action)
                    logp_buf.append(logp)
                    values_buf.append(value_agents.T)
                    rewards_buf.append(reward.T)
                    dones_buf.append(done_array.T)
                    world_buf.append(world_flat)

                obs_arr = jnp.stack(obs_buf)
                actions_arr = jnp.stack(actions_buf)
                logp_arr = jnp.stack(logp_buf)
                values_arr = jnp.stack(values_buf)
                rewards_arr = jnp.stack(rewards_buf)
                dones_arr = jnp.stack(dones_buf)
                world_arr = jnp.stack(world_buf)

                last_world = build_world_state(obs)
                last_world_repeat = jnp.repeat(last_world[:, None, ...], num_agents, axis=1)
                last_world_flat = flatten_obs(last_world_repeat)
                last_value = critic_state.apply_fn(critic_state.params, last_world_flat)
                last_value = last_value.reshape((num_agents, num_envs))

                advantages_all, returns_all = compute_gae(
                    rewards_arr.reshape((num_steps, -1)),
                    values_arr.reshape((num_steps, -1)),
                    dones_arr.reshape((num_steps, -1)),
                    last_value.reshape((-1,)),
                    float(config["GAMMA"]),
                    float(config["GAE_LAMBDA"]),
                )

                for agent_idx in range(num_agents):
                    agent_obs = obs_arr[:, agent_idx]
                    agent_actions = actions_arr[:, agent_idx]
                    agent_logp = logp_arr[:, agent_idx]
                    agent_values = values_arr[:, agent_idx]
                    agent_rewards = rewards_arr[:, agent_idx]
                    agent_dones = dones_arr[:, agent_idx]

                    advantages = advantages_all.reshape((num_steps, num_agents, num_envs))[:, agent_idx]
                    returns = returns_all.reshape((num_steps, num_agents, num_envs))[:, agent_idx]

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
                            new_state, _ = update_actor(
                                actor_state[agent_idx],
                                mbatch,
                                float(config["CLIP_EPS"]),
                                float(config["ENT_COEF"]),
                            )
                            actor_state[agent_idx] = new_state

                world_flat = world_arr.reshape((-1, *world_arr.shape[2:]))
                returns_flat = returns_all.reshape((-1,))
                for _ in range(int(config["UPDATE_EPOCHS"])):
                    critic_state, _ = update_value(critic_state, world_flat, returns_flat)

                if log_enabled:
                    metrics = finalize_info_stats(info_stats)
                    metrics["train/reward_mean"] = float(jnp.mean(jnp.stack(rewards_buf)))
                    metrics["update_step"] = update_step + 1
                    metrics["env_step"] = (update_step + 1) * num_steps * num_envs
                    wandb.log(metrics, step=metrics["env_step"])

            return actor_state, critic_state

        def _log_callback(metrics):
            if log_enabled:
                wandb.log(metrics, step=int(metrics["env_step"]))

        def _env_step(carry, _):
            actor_state, critic_state, env_state, last_obs, rng = carry
            rng, action_rng, step_rng = jax.random.split(rng, 3)

            obs_batch = flatten_obs(last_obs)
            dist = actor_state.apply_fn(actor_state.params, obs_batch)
            action = dist.sample(seed=action_rng)
            logp = dist.log_prob(action)
            env_actions = unflatten_actions(action, num_envs, num_agents)

            world_state = build_world_state(last_obs)
            world_repeat = jnp.repeat(world_state[:, None, ...], num_agents, axis=1)
            world_flat = flatten_obs(world_repeat)
            value = critic_state.apply_fn(critic_state.params, world_flat)

            step_keys = jax.random.split(step_rng, num_envs)
            obs, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_keys, env_state, env_actions)

            done_array = _done_dict_to_array(done, env.agents)
            info_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x), info)

            transition = (
                obs_batch,
                action,
                logp,
                value,
                to_actor_rewards(reward),
                to_actor_dones(done_array),
                world_flat,
                info_mean,
            )
            return (actor_state, critic_state, env_state, obs, rng), transition

        def _update_step(carry, _):
            actor_state, critic_state, env_state, last_obs, rng, update_step = carry
            (actor_state, critic_state, env_state, last_obs, rng), traj = jax.lax.scan(
                _env_step,
                (actor_state, critic_state, env_state, last_obs, rng),
                None,
                length=num_steps,
            )
            (
                obs_arr,
                actions_arr,
                logp_arr,
                values_arr,
                rewards_arr,
                dones_arr,
                world_arr,
                info_means,
            ) = traj

            last_world = build_world_state(last_obs)
            last_world_repeat = jnp.repeat(last_world[:, None, ...], num_agents, axis=1)
            last_world_flat = flatten_obs(last_world_repeat)
            last_value = critic_state.apply_fn(critic_state.params, last_world_flat)

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
                actor_state, critic_state, rng = carry
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, batch_size)

                def _minibatch(state, idx):
                    actor_state, critic_state = state
                    start = idx * minibatch_size
                    mb_idx = perm[start:start + minibatch_size]
                    mbatch = PPOBatch(
                        obs=batch.obs[mb_idx],
                        actions=batch.actions[mb_idx],
                        old_log_probs=batch.old_log_probs[mb_idx],
                        advantages=batch.advantages[mb_idx],
                        returns=batch.returns[mb_idx],
                    )
                    actor_state, _ = update_actor(
                        actor_state,
                        mbatch,
                        float(config["CLIP_EPS"]),
                        float(config["ENT_COEF"]),
                    )
                    critic_state, _ = update_value(
                        critic_state,
                        world_arr.reshape((-1, *world_arr.shape[2:]))[mb_idx],
                        batch.returns[mb_idx],
                    )
                    return (actor_state, critic_state), None

                (actor_state, critic_state), _ = jax.lax.scan(
                    _minibatch,
                    (actor_state, critic_state),
                    jnp.arange(int(config["NUM_MINIBATCHES"])),
                )
                return (actor_state, critic_state, rng), None

            (actor_state, critic_state, rng), _ = jax.lax.scan(
                _update_epoch,
                (actor_state, critic_state, rng),
                None,
                length=int(config["UPDATE_EPOCHS"]),
            )

            info_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), info_means)
            metrics = {
                "train/reward_mean": jnp.mean(rewards_arr),
                "update_step": update_step + 1,
                "env_step": (update_step + 1) * num_steps * num_envs,
            }
            for key in info_mean:
                metrics[f"env/{key}"] = info_mean[key]
            jax.debug.callback(_log_callback, metrics)

            return (actor_state, critic_state, env_state, last_obs, rng, update_step + 1), metrics

        init_carry = (actor_state, critic_state, env_state, obs, rng, 0)

        def _train_shared(carry):
            return jax.lax.scan(_update_step, carry, None, length=num_updates)

        final_carry, _ = jax.jit(_train_shared)(init_carry)
        return final_carry[0], final_carry[1]

    return train
