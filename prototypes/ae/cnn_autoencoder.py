# https://www.kaggle.com/ljlbarbosa/convolution-autoencoder-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.autoencoders.base_autoencoder import BaseAutoencoder


# define the NN architecture
class ConvAutoencoder(BaseAutoencoder):
    def __init__(self):
        super(BaseAutoencoder, self).__init__()
        super(ConvAutoencoder, self).__init__()

        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv2(x))

        return x


if __name__ == "__main__":
    random_data = torch.rand((1, 3, 64, 64))
    print(random_data.shape)

    # initialize the NN
    model = ConvAutoencoder()

    result = model(random_data)
    # print(result)
