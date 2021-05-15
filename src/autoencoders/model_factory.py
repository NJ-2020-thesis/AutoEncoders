import os
import torch
import warnings

from src.utils.model_types import ModelType
from src.autoencoders.autoencoder import AutoEncoder
from src.autoencoders.custom_layers import DefaultEncoder, DefaultDecoder, \
    DefaultCNNEncoder, DefaultCNNDecoder


class ModelFactory:
    def __init__(self, config_values):
        self.model = None
        self.data_loaded = config_values

    def get_model(self, model_type: ModelType):

        # Device on which to run.
        if torch.cuda.is_available():
            device = "cuda"
        else:
            warnings.warn(
                "Please note that although executing on CPU is supported,"
                + "the training is unlikely to finish in reasonable time."
            )
            device = "cpu"

        if model_type == ModelType.AE:
            self.model = AutoEncoder(DefaultEncoder(input_shape=self.data_loaded['model']['encoder']['input_shape'],
                                                    output_shape=self.data_loaded['model']['representation_size']),
                                     DefaultDecoder(input_shape=self.data_loaded['model']['representation_size'],
                                                    output_shape=self.data_loaded['model']['decoder'][
                                                        'input_shape'])).to(device)
        elif model_type == ModelType.CNN_AE:
            self.model = AutoEncoder(
                DefaultCNNEncoder(input_channels=self.data_loaded['model']['encoder']['input_channels'],
                                  output_shape=self.data_loaded['model']['representation_size']),
                DefaultCNNDecoder(output_channels=self.data_loaded['model']['decoder']['output_channels'],
                                  input_shape=self.data_loaded['model']['representation_size'])).to(device)
        elif model_type == ModelType.VAE:
            pass
        elif model_type == ModelType.DSAE:
            pass

        return self.model
