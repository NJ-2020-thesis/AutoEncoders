# https://gist.github.com/AFAgarap/4f8a8d8edf352271fa06d85ba0361f26

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg', force=True)

import sys
sys.path.insert(1, '/home/anaras2s/NJ-2020-thesis/AutoEncoders')

from src.autoencoders.basic_autoencoder import AutoEncoder
from src.dataset_utils.vm_dataset import VisuomotorDataset

MODEL_SAVE = "/home/anaras2s/model/AE/" \
             "gpu_prototype.pth"
INPUT_SHAPE = (28, 28)
INPUT_DIM = 28*28

model = AutoEncoder(input_shape=INPUT_DIM,output_shape=16)
model.load_state_dict(torch.load(MODEL_SAVE,map_location=torch.device('cpu')))
model.eval()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

DATASET_PATH = "/home/anaras2s/anirudh/main_dataset/door_8/*.png"
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