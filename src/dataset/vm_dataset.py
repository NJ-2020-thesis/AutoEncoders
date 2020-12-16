# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
# https://github.com/utkuozbulak/pytorch-custom-dataset-examples

from __future__ import print_function, division
import os
import torch
import glob
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils
from os import path

# Ignore warnings
import warnings
from PIL import Image

warnings.filterwarnings("ignore")


class VisuomotorDataset(Dataset):
    """ Visuomotor Dataset Class"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = glob.glob(root_dir,recursive=True)

    def __getitem__(self, idx):
        current_img_path = self.dataset[idx]

        sample = Image.open(current_img_path)

        name = os.path.basename(current_img_path)
        label = 0 if (name.split("_")[1] == "failure") else 1

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    PATH = "/home/anirudh/Desktop/main_dataset/door_1/*.png"
    vm_dataset = VisuomotorDataset(PATH)
