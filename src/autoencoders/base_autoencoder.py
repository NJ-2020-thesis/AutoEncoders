import torch.nn as nn

class BaseAutoencoder(nn.Module):

    def __init__(self):
        self.activation = {}

    def get_activation(self,name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook


