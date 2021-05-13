import torch

from utils.name_list import *


class BaseAutoencoder(Module):

    def encode(self, x: Tensor):
        pass

    def decode(self, x: Tensor):
        pass
