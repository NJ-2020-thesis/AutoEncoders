import torch

from src.dataset_utils.model_types import ModelType
from src.autoencoders.autoencoder import AutoEncoder
from src.autoencoders.layer_utils import DefaultEncoder, DefaultDecoder, \
    DefaultCNNEncoder, DefaultCNNDecoder


class ModelFactory:
    def __init__(self):
        self.model = None

    def get_model(self, model_type: ModelType):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == ModelType.AE:
            self.model = AutoEncoder(DefaultEncoder(input_shape=625, output_shape=8),
                                     DefaultDecoder(input_shape=8, output_shape=625)).to(device)
        elif model_type == ModelType.CNN_AE:
            self.model = AutoEncoder(DefaultCNNEncoder(input_channels=3, output_shape=8),
                                     DefaultCNNDecoder(output_channels=3, input_shape=8)).to(device)
        elif model_type == ModelType.VAE:
            pass
        elif model_type == ModelType.DSAE:
            pass

        return self.model
