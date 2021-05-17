# https://github.com/utkuozbulak/pytorch-custom-dataset-examples

from __future__ import print_function, division

import glob
import os

# Ignore warnings
from PIL import Image
from torch.utils.data.dataset import Dataset


class VisuomotorDataset(Dataset):
    """
    Visuomotor Dataset Class
    Images extracted from a Kinova3 arm in CoppeliaSim environment.
    """

    def __init__(self, root_dir, transform=None, resize=(64, 64)):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = glob.glob(root_dir, recursive=True)
        self.resize = resize

    def __getitem__(self, idx):
        # self.dataset = torch.utils.data.ConcatDataset([self.dataset, self.dataset])
        current_img_path = self.dataset[idx]

        sample = Image.open(current_img_path).convert("RGB")
        sample = sample.resize(self.resize)

        # label 1 == goal images
        # label 0 == traj/failed images
        name = os.path.basename(current_img_path)
        label = 0 if (name.split("_")[1] == "failure") else 1

        if self.transform:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.dataset)
