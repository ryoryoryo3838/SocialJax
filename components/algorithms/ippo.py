"""Independent PPO implementation for SocialJax environments."""
from __future__ import annotations

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

import socialjax
from socialjax.wrappers.baselines import LogWrapper

from components.algorithms.networks import ActorCritic, EncoderConfig
from components.training.ppo import PPOBatch, compute_gae, update_ppo
from components.training.utils import (
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
        # INIT NETWORK
        if parameter_sharing:
            network = ActorCritic(env.action_space().n, encoder_cfg)
        else:
            network = [ActorCritic(env.action_space().n, encoder_cfg) for _ in range(num_agents)]

        rng, init_rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))

        if parameter_sharing:
            params = network.init(init_rng, init_x)
        else:
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

        if parameter_sharing:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=params,
                tx=tx,
            )
        else:
            train_state = [
                TrainState.create(
                    apply_fn=network[i].apply,
                    params=params[i],
                    tx=tx,
                )
                for i in range(num_agents)
            ]

        # INIT ENV
        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)

        for update_step in range(num_updates):
            obs_buf = []
            actions_buf = []
            logp_buf = []
            values_buf = []
            rewards_buf = []
            dones_buf = []

            for _ in range(num_steps):
                if parameter_sharing:
                    obs_batch = flatten_obs(obs)
                    rng, action_rng = jax.random.split(rng)
                    pi, value = network.apply(train_state.params, obs_batch)
                    action = pi.sample(seed=action_rng)
                    logp = pi.log_prob(action)
                    env_actions = unflatten_actions(action, num_envs, num_agents)
                else:
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

                if parameter_sharing:
                    obs_buf.append(obs_batch)
                    actions_buf.append(action)
                    logp_buf.append(logp)
                    values_buf.append(value)
                    rewards_buf.append(to_actor_rewards(reward))
                    dones_buf.append(to_actor_dones(done_array))
                else:
                    obs_buf.append(jnp.stack(obs_batch, axis=0))
                    actions_buf.append(action)
                    logp_buf.append(logp)
                    values_buf.append(value)
                    rewards_buf.append(reward.T)
                    dones_buf.append(done_array.T)

            if parameter_sharing:
                obs_arr = jnp.stack(obs_buf)
                actions_arr = jnp.stack(actions_buf)
                logp_arr = jnp.stack(logp_buf)
                values_arr = jnp.stack(values_buf)
                rewards_arr = jnp.stack(rewards_buf)
                dones_arr = jnp.stack(dones_buf)

                last_obs = flatten_obs(obs)
                _, last_value = network.apply(train_state.params, last_obs)

                advantages, returns = compute_gae(
                    rewards_arr,
                    values_arr,
                    dones_arr,
                    last_value,
                    float(config["GAMMA"]),
                    float(config["GAE_LAMBDA"]),
                )

                # flatten for minibatch update
                batch = PPOBatch(
                    obs=obs_arr.reshape((-1, *obs_arr.shape[2:])),
                    actions=actions_arr.reshape((-1,)),
                    old_log_probs=logp_arr.reshape((-1,)),
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
                        train_state, _ = update_ppo(
                            train_state,
                            mbatch,
                            float(config["CLIP_EPS"]),
                            float(config["ENT_COEF"]),
                            float(config["VF_COEF"]),
                        )
            else:
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

        return train_state

    return train
