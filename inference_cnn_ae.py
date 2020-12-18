# https://gist.github.com/AFAgarap/4f8a8d8edf352271fa06d85ba0361f26

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg',warn=False, force=True)

from src.autoencoders.cnn_autoencoder import ConvAutoencoder
from src.dataset.vm_dataset import VisuomotorDataset

MODEL_SAVE = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/" \
             "AutoEncoders/model/cnn_ae_test.pth"


model = ConvAutoencoder()
model.load_state_dict(torch.load(MODEL_SAVE))
model.eval()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

DATASET_PATH = "/home/anirudh/Desktop/main_dataset/door_1/*.png"
test_dataset = VisuomotorDataset(DATASET_PATH,transform,(64,64))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=10, shuffle=False
)

test_examples = None

with torch.no_grad():
    for batch_features in test_loader:
        batch_features = batch_features[0]
        test_examples = batch_features
        reconstruction = model(batch_features)
        break

with torch.no_grad():
    number = 10
    plt.figure(figsize=(20, 4))
    for index in range(number):
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(test_examples[index].numpy().reshape(64, 64, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        plt.imshow(reconstruction[index].numpy().reshape(64, 64, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()