from src.dataset_utils.model_types import ModelType
from src.autoencoders.autoencoder import AutoEncoder
from src.autoencoders.layer_utils import DefaultEncoder, DefaultDecoder, \
                                                        DefaultCNNEncoder, DefaultCNNDecoder

import cv2
import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg',warn=False, force=True)


def get_image_representation(model_type:ModelType,
                             model_path:str,
                             img:np.array):

    model = None
    if model_type == ModelType.AE:
        model = AutoEncoder(DefaultEncoder(input_shape=625, output_shape=8),
                            DefaultDecoder(input_shape=8, output_shape=625))
    elif model_type == ModelType.CNN_AE:
        model = AutoEncoder(DefaultCNNEncoder(input_channels=3, output_shape=8),
                            DefaultCNNDecoder(output_channels=3, input_shape=8))

    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output_img, representation = model(img)

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
