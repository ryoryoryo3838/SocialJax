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
from components.training.checkpoint import save_checkpoint
from components.training.logging import init_wandb, log_metrics
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
        encoder_type=config.get("ENCODER_TYPE", "cnn"),
        transformer_patch_size=int(config.get("TRANSFORMER_PATCH_SIZE", 4)),
        transformer_layers=int(config.get("TRANSFORMER_LAYERS", 2)),
        transformer_heads=int(config.get("TRANSFORMER_HEADS", 4)),
        transformer_mlp_dim=int(config.get("TRANSFORMER_MLP_DIM", 128)),
        transformer_embed_dim=int(config.get("TRANSFORMER_EMBED_DIM", 64)),
    )


def make_train(config: Dict):
    env = socialjax.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))
    env = MAPPOWorldStateWrapper(env)
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

    encoder_cfg = _build_encoder_cfg(config)

    def train(rng):
        wandb = init_wandb(config)
        log_enabled = wandb is not None
        ckpt_dir = config.get("CHECKPOINT_DIR")
        ckpt_every = int(config.get("CHECKPOINT_EVERY", 0))
        ckpt_keep = int(config.get("CHECKPOINT_KEEP", 3))

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
            actor_params = jax.tree_util.tree_map(
                lambda x: jnp.stack([x] * num_agents, axis=0), actor_params
            )
            actor_state = TrainState.create(
                apply_fn=actor_net.apply,
                params=actor_params,
                tx=actor_tx,
            )
            actor_state = actor_state.replace(
                step=jnp.zeros((num_agents,), dtype=jnp.int32)
            )

        critic_state = TrainState.create(
            apply_fn=critic_net.apply,
            params=critic_params,
            tx=critic_tx,
        )

        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)

        if not parameter_sharing:
            # Independent policy path fully in JAX, including per-agent updates.
            def _log_callback(metrics):
                log_metrics(metrics, wandb if log_enabled else None)

            def _save_callback(step, actor_params, critic_params, do_save):
                if not ckpt_dir or ckpt_every <= 0 or not do_save:
                    return
                actor_params_list = [
                    jax.tree_util.tree_map(lambda x, i=i: x[i], actor_params)
                    for i in range(num_agents)
                ]
                save_checkpoint(
                    ckpt_dir,
                    int(step),
                    {"actor_params": actor_params_list, "critic_params": critic_params},
                    keep=ckpt_keep,
                )

            def _env_step(carry, _):
                actor_state, critic_state, env_state, last_obs, rng = carry
                rng, action_rng, step_rng = jax.random.split(rng, 3)
                obs_agents = jnp.swapaxes(last_obs, 0, 1)
                agent_rngs = jax.random.split(action_rng, num_agents)
                dist = jax.vmap(actor_state.apply_fn, in_axes=(0, 0))(
                    actor_state.params, obs_agents
                )
                action = jax.vmap(lambda d, k: d.sample(seed=k))(dist, agent_rngs)
                logp = dist.log_prob(action)
                env_actions = list(action)

                world_state = build_world_state(last_obs)
                world_repeat = jnp.repeat(world_state[:, None, ...], num_agents, axis=1)
                world_flat = flatten_obs(world_repeat)
                value = critic_state.apply_fn(critic_state.params, world_flat)
                value_agents = value.reshape((num_agents, num_envs))

                step_keys = jax.random.split(step_rng, num_envs)
                obs, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_keys, env_state, env_actions)

                done_array = _done_dict_to_array(done, env.agents)
                info_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x), info)

                transition = (
                    obs_agents,
                    action,
                    logp,
                    value_agents,
                    reward.T,
                    done_array.T,
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
                last_value = last_value.reshape((num_agents, num_envs))

                rewards_agents = jnp.swapaxes(rewards_arr, 0, 1)
                values_agents = jnp.swapaxes(values_arr, 0, 1)
                dones_agents = jnp.swapaxes(dones_arr, 0, 1)

                advantages, returns = jax.vmap(
                    compute_gae, in_axes=(0, 0, 0, 0, None, None)
                )(
                    rewards_agents,
                    values_agents,
                    dones_agents,
                    last_value,
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
                    return jax.vmap(lambda x_i, idx_i: x_i[idx_i])(x, idx)

                def _update_epoch(carry, _):
                    actor_state, rng = carry
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
                        state, _ = jax.vmap(
                            update_actor, in_axes=(0, 0, None, None)
                        )(
                            state,
                            mbatch,
                            float(config["CLIP_EPS"]),
                            float(config["ENT_COEF"]),
                        )
                        return state, None

                    state, _ = jax.lax.scan(
                        _minibatch,
                        actor_state,
                        jnp.arange(num_minibatches),
                    )
                    return (state, rng), None

                (actor_state, rng), _ = jax.lax.scan(
                    _update_epoch,
                    (actor_state, rng),
                    None,
                    length=int(config["UPDATE_EPOCHS"]),
                )

                world_flat = world_arr.reshape((-1, *world_arr.shape[2:]))
                returns_time = jnp.swapaxes(returns, 0, 1)
                returns_flat = returns_time.reshape((-1,))

                def _critic_epoch(state, _):
                    state, _ = update_value(state, world_flat, returns_flat)
                    return state, None

                critic_state, _ = jax.lax.scan(
                    _critic_epoch,
                    critic_state,
                    None,
                    length=int(config["UPDATE_EPOCHS"]),
                )

                info_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), info_means)
                metrics = {
                    "train/reward_mean": jnp.mean(rewards_arr),
                    "update_step": update_step + 1,
                    "env_step": (update_step + 1) * num_steps * num_envs,
                }
                reward_per_agent = jnp.mean(rewards_arr, axis=(0, 2))
                for agent_idx, value in enumerate(reward_per_agent):
                    metrics[f"agent/{agent_idx}/reward_mean"] = value
                for key in info_mean:
                    metrics[f"env/{key}"] = info_mean[key]
                do_save = (update_step + 1) % ckpt_every == 0 if (ckpt_dir and ckpt_every > 0) else False
                jax.debug.callback(_log_callback, metrics)
                jax.debug.callback(
                    _save_callback,
                    update_step + 1,
                    actor_state.params,
                    critic_state.params,
                    do_save,
                )

                return (actor_state, critic_state, env_state, last_obs, rng, update_step + 1), metrics

            init_carry = (actor_state, critic_state, env_state, obs, rng, 0)

            def _train_independent(carry):
                return jax.lax.scan(_update_step, carry, None, length=num_updates)

            final_carry, _ = jax.jit(_train_independent)(init_carry)
            return final_carry[0], final_carry[1]

        def _log_callback(metrics):
            log_metrics(metrics, wandb if log_enabled else None)

        def _save_callback(step, actor_params, critic_params, do_save):
            if not ckpt_dir or ckpt_every <= 0 or not do_save:
                return
            save_checkpoint(
                ckpt_dir,
                int(step),
                {"actor_params": actor_params, "critic_params": critic_params},
                keep=ckpt_keep,
            )

        def _env_step(carry, _):
            # Collect one environment step with centralized critic state.
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
            # Rollout + update block for a single PPO iteration.
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
                    mb_idx = jax.lax.dynamic_slice(
                        perm,
                        (start,),
                        (minibatch_size,),
                    )
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
            do_save = (update_step + 1) % ckpt_every == 0 if (ckpt_dir and ckpt_every > 0) else False
            jax.debug.callback(_log_callback, metrics)
            jax.debug.callback(
                _save_callback,
                update_step + 1,
                actor_state.params,
                critic_state.params,
                do_save,
            )

            return (actor_state, critic_state, env_state, last_obs, rng, update_step + 1), metrics

        init_carry = (actor_state, critic_state, env_state, obs, rng, 0)

        def _train_shared(carry):
            return jax.lax.scan(_update_step, carry, None, length=num_updates)

        final_carry, _ = jax.jit(_train_shared)(init_carry)
        return final_carry[0], final_carry[1]

    return train
