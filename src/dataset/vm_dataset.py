# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
# https://github.com/utkuozbulak/pytorch-custom-dataset-examples

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")



class VisuomotorDataset(Dataset):
    """ Visuomotor Dataset Class"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = []

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = None
        img = []
        label = ""

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def __len__(self):
        return len(self.dataset)
