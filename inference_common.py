from src.autoencoders.lange_AE import LangeConvAutoencoder
from src.autoencoders.spatial_autoencoder import DeepSpatialAutoencoder
from src.autoencoders.vae_autoencoder import VAE

import cv2
import torch
import numpy as np
from enum import Enum

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg',warn=False, force=True)


class ModelType(Enum):
    LangeCNN = 1
    SpatialAE = 2
    VAE = 3


def get_image_representation(model_type:ModelType, model_path:str,
                             img:np.array,size=(64,64)):
    resized_image = torch.rand((1, 3, 64, 64))

    model = None
    if model_type == ModelType.LangeCNN:
        model = LangeConvAutoencoder()
        model.load_state_dict(torch.load(model_path))
        model.eval()

    if model_type == ModelType.SpatialAE:
        model = DeepSpatialAutoencoder()
        model.load_state_dict(torch.load(model_path))
        model.eval()

    if model_type == ModelType.VAE:
        model = VAE(None,None)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    with torch.no_grad():
        output_img, representation = model(resized_image)

    return output_img, representation


if __name__=="__main__":

    AE_MODEL_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model/" \
                 "lange_vae_18_150_gpu.pth"
    LANGE_MODEL_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model/lange_ae/" \
                       "lange_vae_14_300_gpu.pth"
    VAE_MODEL_PATH = ""

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output, repr_vec = get_image_representation(ModelType.LangeCNN,
                                                LANGE_MODEL_PATH,
                                                None)
    print(output.shape,repr_vec.shape)
