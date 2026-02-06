"""Networks for World Model and RND."""
from __future__ import annotations

from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
import numpy as np

from components.algorithms.networks import EncoderConfig, _build_encoder


class RNDNetwork(nn.Module):
    encoder_cfg: EncoderConfig
    output_dim: int = 64

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        embedding = _build_encoder(self.encoder_cfg, obs.shape[1:])(obs)
        # MLP head
        x = nn.Dense(
            features=self.encoder_cfg.cnn_dense_size,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(embedding)
        x = nn.relu(x)
        x = nn.Dense(
            features=self.output_dim,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(x)
        return x


class WorldModel(nn.Module):
    encoder_cfg: EncoderConfig
    num_agents: int
    action_dim: int
    latent_dim: int = 64

    @nn.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predicts next latent state and rewards.
        
        Args:
            obs: [B, H, W, C] observations
            actions: [B, N] joint actions
        """
        # Encode current observation
        embedding = _build_encoder(self.encoder_cfg, obs.shape[1:])(obs)
        
        # Embed actions
        # actions is [B, N] where N is num agents
        action_one_hot = jax.nn.one_hot(actions, self.action_dim)  # [B, N, D]
        action_flat = action_one_hot.reshape((action_one_hot.shape[0], -1))  # [B, N * D]
        
        # Combine state and action
        combined = jnp.concatenate([embedding, action_flat], axis=-1)
        
        # Predict next latent
        # We use a simple MLP to predict the next embedding
        next_latent = nn.Dense(
            features=self.latent_dim,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(combined)
        next_latent = nn.relu(next_latent)
        next_latent = nn.Dense(
            features=self.latent_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(next_latent)
        
        # Predict rewards (per agent)
        reward = nn.Dense(
            features=self.num_agents,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(combined)
        
        return next_latent, reward


class LatentEncoder(nn.Module):
    """Encodes observation into the same latent space as WorldModel's next_latent."""
    encoder_cfg: EncoderConfig
    latent_dim: int = 64

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        embedding = _build_encoder(self.encoder_cfg, obs.shape[1:])(obs)
        x = nn.Dense(
            features=self.latent_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(embedding)
        return x
