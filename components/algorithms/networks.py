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


def _build_encoder(cfg: EncoderConfig, obs_shape: Sequence[int], name: str = "encoder") -> nn.Module:
    if _is_image_obs(obs_shape):
        if cfg.encoder_type == "transformer":
            return TransformerEncoder(
                patch_size=cfg.transformer_patch_size,
                num_layers=cfg.transformer_layers,
                num_heads=cfg.transformer_heads,
                mlp_dim=cfg.transformer_mlp_dim,
                embed_dim=cfg.transformer_embed_dim,
                activation=cfg.activation,
                name=name,
            )
        return CNNEncoder(
            channels=cfg.cnn_channels,
            kernel_sizes=cfg.cnn_kernel_sizes,
            activation=cfg.activation,
            dense_size=cfg.cnn_dense_size,
            name=name,
        )
    return MLPEncoder(
        hidden_sizes=cfg.mlp_sizes,
        activation=cfg.activation,
        name=name,
    )


class ActorCritic(nn.Module):
    action_dim: int
    encoder_cfg: EncoderConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        embedding = self.encode(x)
        return self.act(embedding)

    @nn.compact
    def encode(self, x: jnp.ndarray):
        return _build_encoder(self.encoder_cfg, x.shape[1:], name="encoder")(x)

    @nn.compact
    def act(self, embedding: jnp.ndarray):
        actor_head = MLPDecoder(
            hidden_sizes=self.encoder_cfg.decoder_hidden_sizes,
            output_size=self.action_dim,
            activation=self.encoder_cfg.activation,
            name="actor_head",
        )
        logits = actor_head(embedding)
        pi = distrax.Categorical(logits=logits)

        critic_head = ValueDecoder(
            hidden_sizes=self.encoder_cfg.decoder_hidden_sizes,
            activation=self.encoder_cfg.activation,
            name="critic_head",
        )
        value = critic_head(embedding)
        return pi, value


class Actor(nn.Module):
    action_dim: int
    encoder_cfg: EncoderConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        embedding = _build_encoder(self.encoder_cfg, x.shape[1:], name="encoder")(x)

        actor_head = MLPDecoder(
            hidden_sizes=self.encoder_cfg.decoder_hidden_sizes,
            output_size=self.action_dim,
            activation=self.encoder_cfg.activation,
            name="actor_head",
        )
        logits = actor_head(embedding)
        return distrax.Categorical(logits=logits)


class Critic(nn.Module):
    encoder_cfg: EncoderConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        embedding = _build_encoder(self.encoder_cfg, x.shape[1:], name="encoder")(x)

        critic_head = ValueDecoder(
            hidden_sizes=self.encoder_cfg.decoder_hidden_sizes,
            activation=self.encoder_cfg.activation,
            name="critic_head",
        )
        return critic_head(embedding)
