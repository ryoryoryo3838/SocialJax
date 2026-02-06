"""Reusable encoders for agents and critics."""
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class MLPEncoder(nn.Module):
    hidden_sizes: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        activation = nn.relu if self.activation == "relu" else nn.tanh
        for size in self.hidden_sizes:
            x = nn.Dense(
                features=size,
                kernel_init=nn.initializers.lecun_normal(),
                bias_init=constant(0.0),
            )(x)
            x = activation(x)
        return x


class CNNEncoder(nn.Module):
    channels: Sequence[int] = (32, 32, 32)
    kernel_sizes: Sequence[Sequence[int]] = ((5, 5), (3, 3), (3, 3))
    activation: str = "relu"
    dense_size: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        activation = nn.relu if self.activation == "relu" else nn.tanh
        for features, kernel_size in zip(self.channels, self.kernel_sizes):
            x = nn.Conv(
                features=features,
                kernel_size=kernel_size,
                kernel_init=nn.initializers.lecun_normal(),
                bias_init=constant(0.0),
            )(x)
            x = activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=self.dense_size,
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        return x


class TransformerEncoder(nn.Module):
    patch_size: int = 4
    num_layers: int = 2
    num_heads: int = 4
    mlp_dim: int = 128
    embed_dim: int = 64
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 4:
            raise ValueError("TransformerEncoder expects image input [B,H,W,C].")
        activation = nn.relu if self.activation == "relu" else nn.tanh

        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=constant(0.0),
        )(x)
        x = x.reshape((x.shape[0], -1, self.embed_dim))

        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, x.shape[1], self.embed_dim),
        )
        x = x + pos_embed

        for _ in range(self.num_layers):
            h = nn.LayerNorm()(x)
            h = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.embed_dim,
                out_features=self.embed_dim,
            )(h, h)
            x = x + h

            h = nn.LayerNorm()(x)
            h = nn.Dense(self.mlp_dim)(h)
            h = activation(h)
            h = nn.Dense(self.embed_dim)(h)
            x = x + h

        x = jnp.mean(x, axis=1)
        return x
