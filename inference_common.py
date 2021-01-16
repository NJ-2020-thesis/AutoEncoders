import cv2

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torch import nn

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg',warn=False, force=True)

from src.autoencoders.lange_AE import ConvAutoencoder
from src.dataset_utils.vm_dataset import VisuomotorDataset


def get_image_representation(model,img,size=(64,64)):
    resized_image = cv2.resize(img,size)
    output, representation = model(resized_image.cuda())

    return output, representation



# plt.imshow(vae(inputs.cuda()).cpu().data[5].numpy().reshape(INPUT_SIZE), cmap='gray')
# plt.show(block=True)
if __name__=="__main__":

    AE_MODEL_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model/" \
                 "lange_vae_18_150_gpu.pth"
    LANGE_MODEL_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model/" \
                 "lange_vae_18_150_gpu.pth"
    VAE_MODEL_PATH = ""

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output, repr_vec = get_image_representation()
    plt.imshow()