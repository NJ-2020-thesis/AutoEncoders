import numpy as np
from typing import List, TypeVar

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.name_list import *


# --------------------MLP Encoder/Decoder-------------------------------


class DefaultEncoder(Module):
    def __init__(self, input_shape: int = 625, output_shape: int = 8):
        super(DefaultEncoder, self).__init__()
        self.enc1 = nn.Linear(in_features=input_shape, out_features=1024)
        self.enc2 = nn.Linear(in_features=1024, out_features=256)
        self.enc3 = nn.Linear(in_features=256, out_features=128)
        self.enc4 = nn.Linear(in_features=128, out_features=64)
        self.enc5 = nn.Linear(in_features=64, out_features=32)
        self.enc6 = nn.Linear(in_features=32, out_features=output_shape)

    def forward(self, x: Tensor):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        x = F.relu(self.enc6(x))

        return x


class DefaultDecoder(Module):
    def __init__(self, input_shape: int = 8, output_shape: int = 625):
        super(DefaultDecoder, self).__init__()
        self.dec1 = nn.Linear(in_features=input_shape, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=1024)
        self.dec6 = nn.Linear(in_features=1024, out_features=output_shape)

    def forward(self, x: Tensor):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        x = F.relu(self.dec6(x))

        return x


# --------------------CNN Encoder/Decoder--------------------------------


class Flatten(Module):
    def forward(self, x: Tensor):
        batch_size = x.size(0)
        return x.view(batch_size, -1)


class DefaultCNNEncoder(Module):
    def __init__(self, input_size: tuple = (64, 64), input_channels: int = 3, output_shape: int = 8):
        super(DefaultCNNEncoder, self).__init__()

        self.flatten = Flatten()  # describing the layer
        self.input_channels = input_channels
        self.input_size = input_size

        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1)

        # Input shape based on image size and output
        # channels from conv layer after flattening
        self.enc_f1 = nn.Linear(in_features=4*input_size[0]*input_size[1], out_features=256)
        self.enc_f2 = nn.Linear(in_features=256, out_features=128)
        self.enc_f3 = nn.Linear(in_features=128, out_features=64)
        self.enc_f4 = nn.Linear(in_features=64, out_features=32)
        self.enc_f5 = nn.Linear(in_features=32, out_features=output_shape)

    def forward(self, x: Tensor):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)

        x = F.relu(self.enc_f1(x))
        x = F.relu(self.enc_f2(x))
        x = F.relu(self.enc_f3(x))
        x = F.relu(self.enc_f4(x))
        x = F.relu(self.enc_f5(x))
        x = self.flatten(x)

        return x


class DefaultCNNDecoder(Module):
    def __init__(self, input_shape: int = 8, output_size: tuple = (64, 64), output_channels: int = 3):
        super(DefaultCNNDecoder, self).__init__()

        self.flatten = Flatten()
        self.output_channels = output_channels
        self.output_size = output_size

        # decoder layers
        self.dec_f1 = nn.Linear(in_features=input_shape, out_features=32)
        self.dec_f2 = nn.Linear(in_features=32, out_features=64)
        self.dec_f3 = nn.Linear(in_features=64, out_features=128)
        self.dec_f4 = nn.Linear(in_features=128, out_features=256)
        self.dec_f5 = nn.Linear(in_features=256, out_features=4*self.output_size[0]*self.output_size[1])

        self.t_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=self.output_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor):
        batch_size = x.size(0)

        x = F.relu(self.dec_f1(x))
        x = F.relu(self.dec_f2(x))
        x = F.relu(self.dec_f3(x))
        x = F.relu(self.dec_f4(x))
        x = F.relu(self.dec_f5(x))

        x = x.view([batch_size, -1, self.output_size[0], self.output_size[1]])

        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))

        return x
