# https://gist.github.com/AFAgarap/4f8a8d8edf352271fa06d85ba0361f26

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg',warn=False, force=True)

from src.autoencoders.basic_autoencoder import AutoEncoder
from src.dataset_utils.vm_dataset import VisuomotorDataset

MODEL_SAVE = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model/" \
             "gpu_ae_prototype.pth"
INPUT_SHAPE = (64, 64)
INPUT_DIM = 64*64

model = AutoEncoder(input_shape=INPUT_DIM,output_shape=10)
model.load_state_dict(torch.load(MODEL_SAVE))
model.eval()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

DATASET_PATH = "/home/anirudh/Desktop/main_dataset/door_5/*.png"
test_dataset = VisuomotorDataset(DATASET_PATH, transform, INPUT_SHAPE)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=10, shuffle=False
)

test_examples = None

with torch.no_grad():
    for batch_features in test_loader:
        batch_features = batch_features[0]
        test_examples = batch_features.view(-1, INPUT_DIM)
        reconstruction = model(test_examples)
        break

with torch.no_grad():
    number = 5
    plt.figure(figsize=(20, 4))
    for index in range(number):
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(test_examples[index].numpy().reshape(INPUT_SHAPE))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        plt.imshow(reconstruction[index].numpy().reshape(INPUT_SHAPE))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()