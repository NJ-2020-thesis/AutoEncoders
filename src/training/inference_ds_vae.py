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
INPUT_SIZE = (64,64)
INPUT_DIM = 64 * 64
batch_size = 35

DATASET_PATH = "/home/anirudh/Desktop/main_dataset/**/*.png"
MODEL_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model" \
                  "/cnn_ds_vae_small_1000_gpu.pth"

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = VisuomotorDataset(DATASET_PATH,transform,INPUT_SIZE)

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

print('Number of samples: ', len(train_dataset))

ds_vae = DeepSpatialAutoencoder(image_output_size=INPUT_SIZE)
ds_vae.load_state_dict(torch.load(MODEL_PATH))
ds_vae.eval()

with torch.no_grad():
    for i, data in enumerate(dataloader, 0):
        inputs, classes = data
        # inputs, classes = inputs.resize_(batch_size, input_dim),classes

print(inputs.shape)

with torch.no_grad():
    forward1 = ds_vae(inputs)
    sample = inputs[0].permute(1, 2, 0)

    print(forward1.shape)
    plt.imshow(sample)

    sample_fwd = forward1[5].permute(1, 2, 0)
    print(sample_fwd.shape)

    plt.imshow(np.squeeze(sample_fwd),cmap='gray')
    plt.show(block=True)

