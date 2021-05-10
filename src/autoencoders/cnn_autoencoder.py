import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.autoencoders.base_autoencoder import BaseAutoencoder
from src.dataset_utils.name_list import *


class CNNAutoencoder(BaseAutoencoder):
    def __init__(self, encoder: Module, decoder: Module,
                 latent_dim: int = 8, **kwargs):
        super(CNNAutoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.latent_dim = latent_dim