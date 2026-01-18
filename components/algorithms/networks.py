"""Shared networks for IPPO/MAPPO/SVO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import distrax
import flax.linen as nn
import jax.numpy as jnp

from components.models.decoder import MLPDecoder, ValueDecoder
from components.models.encoder import CNNEncoder, MLPEncoder, TransformerEncoder


@dataclass(frozen=True)
class EncoderConfig:
    activation: str = "relu"
    mlp_sizes: Sequence[int] = (64, 64)
    cnn_channels: Sequence[int] = (32, 32, 32)
    cnn_kernel_sizes: Sequence[Tuple[int, int]] = ((5, 5), (3, 3), (3, 3))
    cnn_dense_size: int = 64
    encoder_type: str = "cnn"
    transformer_patch_size: int = 4
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_mlp_dim: int = 128
    transformer_embed_dim: int = 64
    decoder_hidden_sizes: Sequence[int] = (64,)


def build_encoder_config(config: Mapping[str, Any]) -> EncoderConfig:
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
        decoder_hidden_sizes=tuple(config.get("DECODER_HIDDEN_SIZES", (64,))),
    )


def _is_image_obs(obs_shape: Sequence[int]) -> bool:
    return len(obs_shape) >= 3


class ActorCritic(nn.Module):
    action_dim: int
    encoder_cfg: EncoderConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if _is_image_obs(x.shape[1:]):
            if self.encoder_cfg.encoder_type == "transformer":
                encoder = TransformerEncoder(
                    patch_size=self.encoder_cfg.transformer_patch_size,
                    num_layers=self.encoder_cfg.transformer_layers,
                    num_heads=self.encoder_cfg.transformer_heads,
                    mlp_dim=self.encoder_cfg.transformer_mlp_dim,
                    embed_dim=self.encoder_cfg.transformer_embed_dim,
                    activation=self.encoder_cfg.activation,
                )
            else:
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
            hidden_sizes=self.encoder_cfg.decoder_hidden_sizes,
            output_size=self.action_dim,
            activation=self.encoder_cfg.activation,
        )
        logits = actor_head(embedding)
        pi = distrax.Categorical(logits=logits)

        critic_head = ValueDecoder(
            hidden_sizes=self.encoder_cfg.decoder_hidden_sizes,
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
            if self.encoder_cfg.encoder_type == "transformer":
                encoder = TransformerEncoder(
                    patch_size=self.encoder_cfg.transformer_patch_size,
                    num_layers=self.encoder_cfg.transformer_layers,
                    num_heads=self.encoder_cfg.transformer_heads,
                    mlp_dim=self.encoder_cfg.transformer_mlp_dim,
                    embed_dim=self.encoder_cfg.transformer_embed_dim,
                    activation=self.encoder_cfg.activation,
                )
            else:
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
            hidden_sizes=self.encoder_cfg.decoder_hidden_sizes,
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
            if self.encoder_cfg.encoder_type == "transformer":
                encoder = TransformerEncoder(
                    patch_size=self.encoder_cfg.transformer_patch_size,
                    num_layers=self.encoder_cfg.transformer_layers,
                    num_heads=self.encoder_cfg.transformer_heads,
                    mlp_dim=self.encoder_cfg.transformer_mlp_dim,
                    embed_dim=self.encoder_cfg.transformer_embed_dim,
                    activation=self.encoder_cfg.activation,
                )
            else:
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
            hidden_sizes=self.encoder_cfg.decoder_hidden_sizes,
            activation=self.encoder_cfg.activation,
        )
        return critic_head(embedding)
