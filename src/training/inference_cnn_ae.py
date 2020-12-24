# https://gist.github.com/AFAgarap/4f8a8d8edf352271fa06d85ba0361f26

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg',warn=False, force=True)

from src.autoencoders.cnn_autoencoder import ConvAutoencoder
from src.dataset_utils.vm_dataset import VisuomotorDataset

MODEL_SAVE = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/" \
             "AutoEncoders/model/cnn_ae_test.pth"


model = ConvAutoencoder()
model.load_state_dict(torch.load(MODEL_SAVE))
model.eval()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

DATASET_PATH = "/home/anirudh/Desktop/main_dataset/**/*.png"
test_dataset = VisuomotorDataset(DATASET_PATH,transform,(64,64))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=10, shuffle=False
)

dataiter = iter(test_loader)
images, labels = dataiter.next()

#Sample outputs
output = model(images)
images = images.numpy()

output = output.view(10, 3, 64, 64)
output = output.detach().numpy()

#Original Images
print("Original Images")
fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True,
                         sharey=True, figsize=(12,4))
for idx in np.arange(5):
    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
    plt.imshow(images[idx].transpose(1,2,0))
plt.show()

#Reconstructed Images
print('Reconstructed Images')
fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True,
                         sharey=True, figsize=(12,4))
for idx in np.arange(5):
    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
    plt.imshow(output[idx].transpose(1,2,0))
plt.show()
