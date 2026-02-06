"""Networks for World Model and RND."""
from __future__ import annotations

from typing import Sequence, Tuple, Union

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
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=constant(0.0),
        )(embedding)
        x = nn.relu(x)
        x = nn.Dense(
            features=self.output_dim,
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=constant(0.0),
        )(x)
        return x


class WorldModel(nn.Module):
    encoder_cfg: EncoderConfig
    num_agents: int
    action_dim: int
    latent_dim: int = 64

    @nn.compact
    def __call__(self, obs: jnp.ndarray, actions: Union[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predicts next latent state and rewards.
        
        Args:
            obs: [B, H, W, C] observations
            actions: [B, N] (int32) OR [B, N, D] (float32 soft-one-hot)
        """
        embedding = self.encode(obs)
        return self.dynamics(embedding, actions)

    @nn.compact
    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        return _build_encoder(self.encoder_cfg, obs.shape[1:])(obs)

    @nn.compact
    def dynamics(self, embedding: jnp.ndarray, actions: Union[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Determine if actions are discrete indices or soft vectors
        if actions.dtype in [jnp.int32, jnp.int64]:
            action_vectors = jax.nn.one_hot(actions, self.action_dim)  # [B, N, D]
        else:
            action_vectors = actions # Assumed [B, N, D]

        action_flat = action_vectors.reshape((action_vectors.shape[0], -1))  # [B, N * D]
        
        # Combine state and action
        combined = jnp.concatenate([embedding, action_flat], axis=-1)
        
        # Predict next latent
        next_latent = nn.Dense(
            features=self.latent_dim,
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=constant(0.0),
        )(combined)
        next_latent = nn.relu(next_latent)
        next_latent = nn.Dense(
            features=self.latent_dim,
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=constant(0.0),
        )(next_latent)
        
        # Predict rewards (per agent)
        reward = nn.Dense(
            features=self.num_agents,
            kernel_init=nn.initializers.lecun_normal(),
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
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=constant(0.0),
        )(embedding)
        return x
