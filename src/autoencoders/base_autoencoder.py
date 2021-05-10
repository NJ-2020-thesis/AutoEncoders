import torch

from dataset_utils.name_list import *


class BaseAutoencoder(Module):

    def encode(self):
        pass

    def decode(self):
        pass


