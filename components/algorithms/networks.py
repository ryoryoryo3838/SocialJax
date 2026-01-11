"""Shared networks for IPPO/MAPPO/SVO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import distrax
import flax.linen as nn
import jax.numpy as jnp

from components.models.decoder import MLPDecoder, ValueDecoder
from components.models.encoder import CNNEncoder, MLPEncoder


@dataclass(frozen=True)
class EncoderConfig:
    activation: str = "relu"
    mlp_sizes: Sequence[int] = (64, 64)
    cnn_channels: Sequence[int] = (32, 32, 32)
    cnn_kernel_sizes: Sequence[Tuple[int, int]] = ((5, 5), (3, 3), (3, 3))
    cnn_dense_size: int = 64


def _is_image_obs(obs_shape: Sequence[int]) -> bool:
    return len(obs_shape) >= 3


class ActorCritic(nn.Module):
    action_dim: int
    encoder_cfg: EncoderConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if _is_image_obs(x.shape[1:]):
            encoder = CNNEncoder(
                channels=self.encoder_cfg.cnn_channels,
                kernel_sizes=self.encoder_cfg.cnn_kernel_sizes,
                activation=self.encoder_cfg.activation,
                dense_size=self.encoder_cfg.cnn_dense_size,
            )
        else:
            encoder = MLPEncoder(
                hidden_sizes=self.encoder_cfg.mlp_sizes,
                activation=self.encoder_cfg.activation,
            )
        embedding = encoder(x)

        actor_head = MLPDecoder(
            hidden_sizes=(64,),
            output_size=self.action_dim,
            activation=self.encoder_cfg.activation,
        )
        logits = actor_head(embedding)
        pi = distrax.Categorical(logits=logits)

        critic_head = ValueDecoder(
            hidden_sizes=(64,),
            activation=self.encoder_cfg.activation,
        )
        value = critic_head(embedding)
        return pi, value


class Actor(nn.Module):
    action_dim: int
    encoder_cfg: EncoderConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if _is_image_obs(x.shape[1:]):
            encoder = CNNEncoder(
                channels=self.encoder_cfg.cnn_channels,
                kernel_sizes=self.encoder_cfg.cnn_kernel_sizes,
                activation=self.encoder_cfg.activation,
                dense_size=self.encoder_cfg.cnn_dense_size,
            )
        else:
            encoder = MLPEncoder(
                hidden_sizes=self.encoder_cfg.mlp_sizes,
                activation=self.encoder_cfg.activation,
            )
        embedding = encoder(x)

        actor_head = MLPDecoder(
            hidden_sizes=(64,),
            output_size=self.action_dim,
            activation=self.encoder_cfg.activation,
        )
        logits = actor_head(embedding)
        return distrax.Categorical(logits=logits)


class Critic(nn.Module):
    encoder_cfg: EncoderConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if _is_image_obs(x.shape[1:]):
            encoder = CNNEncoder(
                channels=self.encoder_cfg.cnn_channels,
                kernel_sizes=self.encoder_cfg.cnn_kernel_sizes,
                activation=self.encoder_cfg.activation,
                dense_size=self.encoder_cfg.cnn_dense_size,
            )
        else:
            encoder = MLPEncoder(
                hidden_sizes=self.encoder_cfg.mlp_sizes,
                activation=self.encoder_cfg.activation,
            )
        embedding = encoder(x)

        critic_head = ValueDecoder(
            hidden_sizes=(64,),
            activation=self.encoder_cfg.activation,
        )
        return critic_head(embedding)
