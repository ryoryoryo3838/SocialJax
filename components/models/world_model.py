"""World Model for Model-Based Reinforcement Learning."""
from typing import Tuple

import flax.linen as nn
from components.models.encoder import CNNEncoder
from components.models.decoder import ImageDecoder, RewardDecoder
from components.models.dynamics import LatentDynamics

class WorldModel(nn.Module):
    action_dim: int
    input_shape: Tuple[int, int, int] # Observation shape
    activation: str = "relu"

    def setup(self):
        self.encoder = CNNEncoder(activation=self.activation)
        self.decoder = ImageDecoder(output_shape=self.input_shape, activation=self.activation)
        self.dynamics = LatentDynamics(activation=self.activation)
        self.reward_model = RewardDecoder(activation=self.activation)

    def __call__(self, obs, action):
        z = self.encoder(obs)
        rec_obs = self.decoder(z)
        z_next = self.dynamics(z, action, self.action_dim)
        r_pred = self.reward_model(z, action, self.action_dim)
        return z, rec_obs, z_next, r_pred

    def get_latent(self, obs):
        return self.encoder(obs)

    def predict_next(self, z, action):
        z_next = self.dynamics(z, action, self.action_dim)
        r_pred = self.reward_model(z, action, self.action_dim)
        return z_next, r_pred
