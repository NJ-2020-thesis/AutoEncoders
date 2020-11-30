# https://analyticsindiamag.com/hands-on-guide-to-implement-deep-autoencoder-in-pytorch-for-image-reconstruction/

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.autoencoders.base_autoencoder import BaseAutoencoder


class AutoEncoder(BaseAutoencoder):
    def __init__(self,**kwargs):
        super(BaseAutoencoder, self).__init__()
        super(AutoEncoder, self).__init__()

        # Encoder
        self.enc1 = nn.Linear(in_features=kwargs["input_shape"], out_features=256)  # Input image (28*28 = 784)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc5 = nn.Linear(in_features=32, out_features=kwargs["output_shape"])

        # Decoder
        self.dec1 = nn.Linear(in_features=kwargs["output_shape"], out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=kwargs["input_shape"])
        # Output image (28*28 = 784)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))

        return x


if __name__ == "__main__":
    random_data = torch.rand((1, 1, 28, 28))
    print(random_data.shape)
    flat_data = torch.flatten(random_data)
    print(flat_data.shape)

    my_nn = AutoEncoder(input_shape=784,output_shape=16)
    my_nn.enc5.register_forward_hook(my_nn.get_activation('enc5'))

    result = my_nn(flat_data)
    print(my_nn.activation['enc5'])
