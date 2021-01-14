# https://www.kaggle.com/ljlbarbosa/convolution-autoencoder-pytorch
# http://rll.berkeley.edu/dsae/dsae.pdf
# http://ml.informatik.uni-freiburg.de/former/_media/publications/rieijcnn12.pdf
# https://datascience.stackexchange.com/questions/40906/determining-size-of-fc-layer-after-conv-layer-in-pytorch
# https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from src.autoencoders.base_autoencoder import BaseAutoencoder
from src.dataset_utils.vm_dataset import VisuomotorDataset


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# define the NN architecture
class ConvAutoencoder(BaseAutoencoder):
    def __init__(self):
        super(BaseAutoencoder, self).__init__()
        super(ConvAutoencoder, self).__init__()
        self.flatten = Flatten()  # describing the layer

        representation_size = 14

        # encoder layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=7, padding=1,stride=2)

        self.enc_f1 = nn.Linear(in_features=2704, out_features=144)
        self.enc_f2 = nn.Linear(in_features=144, out_features=72)
        self.enc_f3 = nn.Linear(in_features=72, out_features=36)
        self.enc_f4 = nn.Linear(in_features=36, out_features=18)
        self.enc_f5 = nn.Linear(in_features=18, out_features=representation_size)

        # decoder layers
        self.dec_f1 = nn.Linear(in_features=representation_size, out_features=18)
        self.dec_f2 = nn.Linear(in_features=18, out_features=36)
        self.dec_f3 = nn.Linear(in_features=36, out_features=72)
        self.dec_f4 = nn.Linear(in_features=72, out_features=144)
        self.dec_f5 = nn.Linear(in_features=144, out_features=2704)

        self.t_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=7, padding=1,stride=2,output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=7,padding=1)
        self.t_conv3 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=7, padding=1)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        unflatten_size = x.shape
        x = self.flatten(x)

        x = F.relu(self.enc_f1(x))
        x = F.relu(self.enc_f2(x))
        x = F.relu(self.enc_f3(x))
        x = F.relu(self.enc_f4(x))
        x = F.relu(self.enc_f5(x))

        ## decode ##
        x = F.relu(self.dec_f1(x))
        x = F.relu(self.dec_f2(x))
        x = F.relu(self.dec_f3(x))
        x = F.relu(self.dec_f4(x))
        x = F.relu(self.dec_f5(x))
        x = x.reshape(unflatten_size)

        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))

        return x


if __name__ == "__main__":
    # random_data = torch.rand((1, 3, 64, 64))
    # print(random_data.shape)
    #
    # # initialize the NN
    # model = ConvAutoencoder()
    # # model.conv2.register_forward_hook(model.get_activation('conv2'))
    # # summary(model, (3, 64, 64))
    # # print(model)
    #
    # result = model(random_data)
    # # print(result)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvAutoencoder()
    vgg = model.to(device)

    summary(vgg, (3, 64, 64))

