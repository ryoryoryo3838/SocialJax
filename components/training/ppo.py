"""Shared PPO utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import distrax
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PPOBatch:
    obs: jnp.ndarray
    actions: jnp.ndarray
    old_log_probs: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE advantages and returns.

    rewards, values, dones: [num_steps, num_actors]
    last_value: [num_actors]
    """
    num_steps = rewards.shape[0]
    advantages = []
    gae = jnp.zeros_like(last_value)

    for t in reversed(range(num_steps)):
        mask = 1.0 - dones[t]
        next_value = last_value if t == num_steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.append(gae)
    advantages = jnp.stack(advantages[::-1])
    returns = advantages + values
    return advantages, returns


def _loss_fn(
    apply_fn: Callable,
    params,
    batch: PPOBatch,
    clip_eps: float,
    ent_coef: float,
    vf_coef: float,
):
    dist, value = apply_fn(params, batch.obs)
    log_probs = dist.log_prob(batch.actions)
    entropy = dist.entropy().mean()

    ratios = jnp.exp(log_probs - batch.old_log_probs)
    unclipped = ratios * batch.advantages
    clipped = jnp.clip(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * batch.advantages
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

    value_loss = jnp.mean(jnp.square(batch.returns - value))
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
    }
    return loss, metrics


def update_ppo(
    train_state,
    batch: PPOBatch,
    clip_eps: float,
    ent_coef: float,
    vf_coef: float,
):
    def _loss(params, batch, clip_eps, ent_coef, vf_coef):
        return _loss_fn(
            train_state.apply_fn,
            params,
            batch,
            clip_eps,
            ent_coef,
            vf_coef,
        )

    grad_fn = jax.value_and_grad(_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(
        train_state.params,
        batch,
        clip_eps,
        ent_coef,
        vf_coef,
    )
    new_state = train_state.apply_gradients(grads=grads)
    metrics = {**metrics, "loss": loss}
    return new_state, metrics


def update_actor(
    train_state,
    batch: PPOBatch,
    clip_eps: float,
    ent_coef: float,
):
    def _actor_loss(params, batch, clip_eps, ent_coef):
        dist = train_state.apply_fn(params, batch.obs)
        log_probs = dist.log_prob(batch.actions)
        entropy = dist.entropy().mean()
        ratios = jnp.exp(log_probs - batch.old_log_probs)
        unclipped = ratios * batch.advantages
        clipped = jnp.clip(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * batch.advantages
        policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
        loss = policy_loss - ent_coef * entropy
        return loss, {"policy_loss": policy_loss, "entropy": entropy}

    grad_fn = jax.value_and_grad(_actor_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(
        train_state.params,
        batch,
        clip_eps,
        ent_coef,
    )
    new_state = train_state.apply_gradients(grads=grads)
    metrics = {**metrics, "loss": loss}
    return new_state, metrics


def update_value(train_state, obs: jnp.ndarray, returns: jnp.ndarray):
    def _value_loss(params, obs, returns):
        value = train_state.apply_fn(params, obs)
        loss = jnp.mean(jnp.square(returns - value))
        return loss

    grad_fn = jax.value_and_grad(_value_loss)
    loss, grads = grad_fn(train_state.params, obs, returns)
    new_state = train_state.apply_gradients(grads=grads)
    return new_state, {"value_loss": loss}
