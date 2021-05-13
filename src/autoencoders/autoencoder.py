import numpy as np
from typing import List, TypeVar

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.autoencoders.base_autoencoder import BaseAutoencoder
from src.utils.name_list import *


class AutoEncoder(BaseAutoencoder):
    def __init__(self, encoder: Module, decoder: Module,
                 latent_dim: int = 8, **kwargs):
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.latent_dim = latent_dim
        self.criterion = nn.MSELoss()

    def encode(self, x: Tensor):
        latent_vec = self.encoder(x)
        return latent_vec

    def decode(self, z: Tensor):
        output = self.decoder(z)
        return output

    def forward(self, x: Tensor):
        latent_vec = self.encode(x)
        x = self.decode(latent_vec)
        return x, latent_vec

    def loss(self, x: Tensor, x_dash: Tensor):
        train_loss = self.criterion(x, x_dash)
        return train_loss
