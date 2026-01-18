"""SVO-enhanced PPO implementation for SocialJax environments."""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
import optax

import socialjax
from socialjax.wrappers.baselines import LogWrapper

from components.algorithms.networks import ActorCritic, build_encoder_config
from components.algorithms.utils import done_dict_to_array, broadcast_agent_leaves, stack_agent_params
from components.shaping.svo import svo_deviation_penalty, svo_linear_combination
from components.training.checkpoint import save_agent_checkpoints
from components.training.logging import init_wandb, log_metrics
from components.training.ppo import PPOBatch, compute_gae, update_ppo, update_ppo_params


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


def make_train(config: Dict):
    env = socialjax.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))
    env = LogWrapper(env, replace_info=False)

    num_envs = int(config["NUM_ENVS"])
    num_steps = int(config["NUM_STEPS"])
    num_agents = env.num_agents
    parameter_sharing = False

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

    encoder_cfg = build_encoder_config(config)

    def _shape_reward(reward: jnp.ndarray):
        if svo_mode == "linear":
            shaped, theta = svo_linear_combination(reward, svo_angles, svo_target_mask)
        else:
            shaped, theta = svo_deviation_penalty(reward, svo_ideal, svo_w, svo_target_mask)
        return shaped, theta

    def train(rng):
        wandb = init_wandb(config)
        log_enabled = wandb is not None
        ckpt_dir = config.get("CHECKPOINT_DIR")
        ckpt_every = int(config.get("CHECKPOINT_EVERY", 0))
        ckpt_keep = int(config.get("CHECKPOINT_KEEP", 3))

        if not parameter_sharing:
            # Independent policy path fully in JAX, including per-agent updates.
            network = ActorCritic(env.action_space().n, encoder_cfg)
            rng, init_rng = jax.random.split(rng)
            init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
            base_params = network.init(init_rng, init_x)
            params = stack_agent_params(base_params, num_agents)

            if config.get("ANNEAL_LR", False):
                def lr_schedule(count):
                    frac = (
                        1.0
                        - (count // (int(config["NUM_MINIBATCHES"]) * int(config["UPDATE_EPOCHS"])))
                        / num_updates
                    )
                    return float(config["LR"]) * frac

                tx = optax.adam(learning_rate=lr_schedule, eps=1e-5)
            else:
                tx = optax.adam(float(config["LR"]), eps=1e-5)

            opt_state = tx.init(params)
            opt_state = broadcast_agent_leaves(opt_state, num_agents)

            rng, reset_rng = jax.random.split(rng)
            reset_keys = jax.random.split(reset_rng, num_envs)
            # Reset envs to start the rollout.
            obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)

            def _log_callback(metrics):
                log_metrics(metrics, wandb if log_enabled else None)

            def _save_callback(step, params, do_save):
                if not ckpt_dir or ckpt_every <= 0 or not do_save:
                    return
                params_list = [
                    {"params": jax.tree_util.tree_map(lambda x, i=i: x[i], params)}
                    for i in range(num_agents)
                ]
                save_agent_checkpoints(ckpt_dir, int(step), params_list, keep=ckpt_keep)

            def _env_step(carry, _):
                params, opt_state, env_state, last_obs, rng = carry
                rng, action_rng, step_rng = jax.random.split(rng, 3)
                obs_agents = jnp.swapaxes(last_obs, 0, 1)
                agent_rngs = jax.random.split(action_rng, num_agents)
                pi, value = jax.vmap(network.apply, in_axes=(0, 0))(params, obs_agents)
                action = jax.vmap(lambda dist, key: dist.sample(seed=key))(pi, agent_rngs)
                logp = pi.log_prob(action)
                env_actions = list(action)

                step_keys = jax.random.split(step_rng, num_envs)
                # Step environments for one rollout timestep.
                obs, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_keys, env_state, env_actions)

                done_array = done_dict_to_array(done, env.agents)
                # Apply SVO reward shaping.
                shaped_reward, theta = _shape_reward(reward)
                info_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x), info)

                transition = (
                    obs_agents,
                    action,
                    logp,
                    value,
                    shaped_reward.T,
                    reward.T,
                    done_array.T,
                    theta,
                    info_mean,
                )
                return (params, opt_state, env_state, obs, rng), transition

            def _update_step(carry, _):
                params, opt_state, env_state, last_obs, rng, update_step = carry
                # Roll out trajectories for NUM_STEPS.
                (params, opt_state, env_state, last_obs, rng), traj = jax.lax.scan(
                    _env_step,
                    (params, opt_state, env_state, last_obs, rng),
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

                last_obs_agents = jnp.swapaxes(last_obs, 0, 1)
                _, last_values = jax.vmap(network.apply, in_axes=(0, 0))(params, last_obs_agents)

                rewards_agents = jnp.swapaxes(rewards_arr, 0, 1)
                values_agents = jnp.swapaxes(values_arr, 0, 1)
                dones_agents = jnp.swapaxes(dones_arr, 0, 1)

                # Compute GAE/returns from rollout.
                advantages, returns = jax.vmap(
                    compute_gae, in_axes=(0, 0, 0, 0, None, None)
                )(
                    rewards_agents,
                    values_agents,
                    dones_agents,
                    last_values,
                    float(config["GAMMA"]),
                    float(config["GAE_LAMBDA"]),
                )

                obs_agents = jnp.swapaxes(obs_arr, 0, 1)
                actions_agents = jnp.swapaxes(actions_arr, 0, 1)
                logp_agents = jnp.swapaxes(logp_arr, 0, 1)

                batch = PPOBatch(
                    obs=obs_agents.reshape((num_agents, -1, *obs_agents.shape[3:])),
                    actions=actions_agents.reshape((num_agents, -1)),
                    old_log_probs=logp_agents.reshape((num_agents, -1)),
                    advantages=advantages.reshape((num_agents, -1)),
                    returns=returns.reshape((num_agents, -1)),
                )

                batch_size = batch.actions.shape[1]
                num_minibatches = int(config["NUM_MINIBATCHES"])

                def _gather(x, idx):
                    x = jnp.asarray(x)
                    if x.ndim == 0:
                        x = jnp.broadcast_to(x, (num_agents, batch_size))
                    elif x.ndim == 1:
                        x = jnp.broadcast_to(x[None, :], (num_agents, x.shape[0]))
                    take_idx = idx[(...,) + (None,) * (x.ndim - 2)]
                    return jnp.take_along_axis(x, take_idx, axis=1)

                def _update_epoch(carry, _):
                    state, rng = carry
                    rng, perm_rng = jax.random.split(rng)
                    agent_rngs = jax.random.split(perm_rng, num_agents)
                    perm = jax.vmap(
                        lambda key: jax.random.permutation(key, batch_size)
                    )(agent_rngs)
                    perm = perm.reshape((num_agents, num_minibatches, minibatch_size))

                    def _minibatch(state, idx):
                        mb_idx = perm[:, idx]
                        mbatch = PPOBatch(
                            obs=_gather(batch.obs, mb_idx),
                            actions=_gather(batch.actions, mb_idx),
                            old_log_probs=_gather(batch.old_log_probs, mb_idx),
                            advantages=_gather(batch.advantages, mb_idx),
                            returns=_gather(batch.returns, mb_idx),
                        )
                        params, opt_state = state
                        agent_ids = jnp.arange(num_agents)

                        def _update_agent(agent_idx):
                            p = jax.tree_util.tree_map(lambda x: x[agent_idx], params)
                            o = jax.tree_util.tree_map(lambda x: x[agent_idx], opt_state)
                            b = PPOBatch(
                                obs=mbatch.obs[agent_idx],
                                actions=mbatch.actions[agent_idx],
                                old_log_probs=mbatch.old_log_probs[agent_idx],
                                advantages=mbatch.advantages[agent_idx],
                                returns=mbatch.returns[agent_idx],
                            )
                            return update_ppo_params(
                                network.apply,
                                p,
                                o,
                                tx,
                                b,
                                float(config["CLIP_EPS"]),
                                float(config["ENT_COEF"]),
                                float(config["VF_COEF"]),
                                max_grad_norm=float(config["MAX_GRAD_NORM"]),
                            )

                        params, opt_state = jax.vmap(_update_agent)(agent_ids)
                        return (params, opt_state), None

                    state, _ = jax.lax.scan(
                        _minibatch,
                        state,
                        jnp.arange(num_minibatches),
                    )
                    return (state, rng), None

                # PPO update over epochs/minibatches.
                ((params, opt_state), rng), _ = jax.lax.scan(
                    _update_epoch,
                    ((params, opt_state), rng),
                    None,
                    length=int(config["UPDATE_EPOCHS"]),
                )

                info_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), info_means)
                # Log metrics and optionally save checkpoints.
                metrics = {
                    "train/svo_reward_mean": jnp.mean(rewards_arr),
                    "train/extrinsic_reward_mean": jnp.mean(extrinsic_arr),
                    "svo/theta_mean": jnp.mean(thetas_arr),
                    "update_step": update_step + 1,
                    "env_step": (update_step + 1) * num_steps * num_envs,
                }
                reward_per_agent = jnp.mean(rewards_arr, axis=(0, 2))
                extrinsic_per_agent = jnp.mean(extrinsic_arr, axis=(0, 2))
                for agent_idx, value in enumerate(reward_per_agent):
                    metrics[f"agent/{agent_idx}/svo_reward_mean"] = value
                for agent_idx, value in enumerate(extrinsic_per_agent):
                    metrics[f"agent/{agent_idx}/extrinsic_reward_mean"] = value
                for key in info_mean:
                    metrics[f"env/{key}"] = info_mean[key]
                do_save = (update_step + 1) % ckpt_every == 0 if (ckpt_dir and ckpt_every > 0) else False
                jax.debug.callback(_log_callback, metrics)
                jax.debug.callback(_save_callback, update_step + 1, params, do_save)

                return (params, opt_state, env_state, last_obs, rng, update_step + 1), metrics

            init_carry = (params, opt_state, env_state, obs, rng, 0)

            def _train_independent(carry):
                return jax.lax.scan(_update_step, carry, None, length=num_updates)

            final_carry, _ = jax.jit(_train_independent)(init_carry)
            return final_carry[0]

    return train
