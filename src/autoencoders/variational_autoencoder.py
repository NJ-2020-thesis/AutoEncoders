# https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py

import numpy as np
from torch.autograd import Variable

from src.utils.name_list import *


class VariationalAutoEncoder(Module):
    def __init__(self, encoder, decoder, latent_dim: int = 8):
        super(VariationalAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

        self._enc_mu = torch.nn.Linear(100, self.latent_dim)
        self._enc_log_sigma = torch.nn.Linear(100, self.latent_dim)

    def encode(self, x: Tensor):
        pass

    def decode(self, x: Tensor):
        pass

    def forward(self, x: Tensor):
        h_enc = self.encoder(x)
        z = self.__sample_latent(h_enc)
        return self.decoder(z), z

    def __sample_latent(self, h_enc):
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        # Reparametrizarion trick
        return mu + sigma * Variable(std_z, requires_grad=False)
