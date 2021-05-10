import numpy as np
from typing import List, TypeVar

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset_utils.name_list import *


class DefaultEncoder(Module):
    def __init__(self, input_shape, output_shape):
        super(DefaultEncoder, self).__init__()
        self.enc1 = nn.Linear(in_features=input_shape, out_features=1024)
        self.enc2 = nn.Linear(in_features=1024, out_features=256)
        self.enc3 = nn.Linear(in_features=256, out_features=128)
        self.enc4 = nn.Linear(in_features=128, out_features=64)
        self.enc5 = nn.Linear(in_features=64, out_features=32)
        self.enc6 = nn.Linear(in_features=32, out_features=output_shape)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        x = F.relu(self.enc6(x))

        return x


class DefaultDecoder(Module):
    def __init__(self, input_shape, output_shape):
        super(DefaultDecoder, self).__init__()
        self.dec1 = nn.Linear(in_features=input_shape, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=1024)
        self.dec6 = nn.Linear(in_features=1024, out_features=output_shape)

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        x = F.relu(self.dec6(x))

        return x

# -----------------------------------------------------------------------


class DefaultCNNEncoder(Module):
    def __init__(self, input_channels: int, output_shape):
        super(DefaultCNNEncoder, self).__init__()
        self.input_channels = input_channels

        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=16, kernel_size=3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation

        return x


class DefaultCNNDecoder(Module):
    def __init__(self, input_channels: int, output_shape):
        super(DefaultCNNDecoder, self).__init__()
        self.input_channels = input_channels

        self.t_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv2(x))

        return x

