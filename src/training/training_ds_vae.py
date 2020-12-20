import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch import nn
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg',warn=False, force=True)

from src.autoencoders.spatial_autoencoder import DeepSpatialAutoencoder, DSAE_Loss
from src.dataset_utils.vm_dataset import VisuomotorDataset

EPOCHS = 100
input_dim = 28 * 28
batch_size = 64

DATASET_PATH = "/home/anirudh/Desktop/main_dataset/**/*.png"
MODEL_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model" \
                  "/cnn_ds_vae_100_test.pth"

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = VisuomotorDataset(DATASET_PATH,transform,(28,28))

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

print('Number of samples: ', len(train_dataset))

ds_vae = DeepSpatialAutoencoder(image_output_size=(28,28))

optimizer = optim.Adam(ds_vae.parameters(), lr=0.0003)




