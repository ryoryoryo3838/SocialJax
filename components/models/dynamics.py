"""Reusable dynamics models."""
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

class LatentDynamics(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, z, action, action_dim):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        
        # One-hot action
        a_onehot = jax.nn.one_hot(action, num_classes=action_dim)
        
        x = jnp.concatenate([z, a_onehot], axis=-1)
        x = nn.Dense(features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        # Residual connection
        return z + x
