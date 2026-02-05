"""Reusable decoders for policy and value heads."""
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class MLPDecoder(nn.Module):
    hidden_sizes: Sequence[int]
    output_size: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        activation = nn.relu if self.activation == "relu" else nn.tanh
        for size in self.hidden_sizes:
            x = nn.Dense(
                features=size,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(x)
            x = activation(x)
        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        return x


class ValueDecoder(nn.Module):
    hidden_sizes: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        activation = nn.relu if self.activation == "relu" else nn.tanh
        for size in self.hidden_sizes:
            x = nn.Dense(
                features=size,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(x)
            x = activation(x)
        x = nn.Dense(
            features=1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)
        return jnp.squeeze(x, axis=-1)


class ImageDecoder(nn.Module):
    output_shape: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        activation = nn.relu if self.activation == "relu" else nn.tanh
        
        # Simple Dense Decoder for now
        flat_size = np.prod(self.output_shape)
        x = nn.Dense(features=flat_size, kernel_init=orthogonal(np.sqrt(2.0)), bias_init=constant(0.0))(z)
        # x = activation(x) # Usually final layer of decoder is raw or sigmoid. Keeping raw (linear) for MSE
        x = x.reshape((x.shape[0], *self.output_shape))
        return x


class RewardDecoder(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, z: jnp.ndarray, action: jnp.ndarray, action_dim: int) -> jnp.ndarray:
        activation = nn.relu if self.activation == "relu" else nn.tanh
            
        a_onehot = jax.nn.one_hot(action, num_classes=action_dim)
        x = jnp.concatenate([z, a_onehot], axis=-1)
        x = nn.Dense(features=64, kernel_init=orthogonal(np.sqrt(2.0)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(features=1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return jnp.squeeze(x, axis=-1)
