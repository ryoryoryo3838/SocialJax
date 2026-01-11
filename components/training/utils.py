"""Training utilities for SocialJax."""
from __future__ import annotations

from typing import List, Tuple

import jax.numpy as jnp


def flatten_obs(obs: jnp.ndarray) -> jnp.ndarray:
    """Flatten obs to [num_actors, ...] with agents first."""
    # obs: [num_envs, num_agents, ...]
    obs = jnp.transpose(obs, (1, 0, *range(2, obs.ndim)))
    return obs.reshape((obs.shape[0] * obs.shape[1], *obs.shape[2:]))


def unflatten_actions(actions: jnp.ndarray, num_envs: int, num_agents: int) -> List[jnp.ndarray]:
    """Convert flat actions to per-agent action arrays for env.step."""
    actions = actions.reshape((num_agents, num_envs))
    return [actions[i] for i in range(num_agents)]


def build_world_state(obs: jnp.ndarray) -> jnp.ndarray:
    """Build centralized state by concatenating agent observations."""
    # obs: [num_envs, num_agents, ...]
    if obs.ndim <= 2:
        return obs
    if obs.ndim == 3:
        # vector obs: [num_envs, num_agents, obs_dim]
        num_envs, num_agents, obs_dim = obs.shape
        return obs.reshape((num_envs, num_agents * obs_dim))

    # image obs: [num_envs, num_agents, H, W, C]
    num_envs, num_agents = obs.shape[:2]
    spatial = obs.shape[2:-1]
    channels = obs.shape[-1]
    world_state = jnp.transpose(obs, (0, 2, 3, 1, 4))
    world_state = world_state.reshape((num_envs, *spatial, channels * num_agents))
    return world_state


def to_actor_obs(obs: jnp.ndarray) -> jnp.ndarray:
    """Prepare actor obs as [num_actors, ...]."""
    return flatten_obs(obs)


def to_actor_rewards(rewards: jnp.ndarray) -> jnp.ndarray:
    """Flatten rewards to [num_actors]."""
    # rewards: [num_envs, num_agents]
    rewards = rewards.transpose((1, 0))
    return rewards.reshape((rewards.shape[0] * rewards.shape[1],))


def to_actor_dones(dones: jnp.ndarray) -> jnp.ndarray:
    """Flatten done flags to [num_actors]."""
    # dones: [num_envs, num_agents]
    dones = dones.transpose((1, 0))
    return dones.reshape((dones.shape[0] * dones.shape[1],))
