from src.autoencoders.autoencoder import AutoEncoder
from src.dataset_utils.model_types import ModelType

import cv2
import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg',warn=False, force=True)

def get_image_representation(model_type:ModelType,
                             model_path:str,
                             img:np.array):
    """
    Creates a representation vector from an input image.
    :param model_type:
    :param model_path:
    :param img:
    :return:
    """

    model = None
    if model_type == ModelType.AE:
        model = AutoEncoder()

    elif model_type == ModelType.LangeCNN:
        pass

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

    resized_image = torch.rand((1, 3, 64, 64))

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output, repr_vec = get_image_representation(ModelType.LangeCNN,
                                                LANGE_MODEL_PATH,
                                                resized_image)
    print(output.shape,repr_vec.shape)
