import torch

import unittest
from src.autoencoders.autoencoder import AutoEncoder
from src.autoencoders.layer_utils import DefaultEncoder, DefaultDecoder, \
                                                        DefaultCNNEncoder, DefaultCNNDecoder


class LayerTest(unittest.TestCase):

    @unittest.skip("Skipping Test")
    def test_mlp_encoder_decoder(self):
        random_input = torch.rand(625)
        random_output = torch.rand(8)

        encoder = DefaultEncoder(output_shape=625, input_shape=8)
        decoder = DefaultDecoder(output_shape=8, input_shape=625)

        assert(encoder(random_input).shape == torch.Size([8]))
        assert(decoder(random_output).shape == torch.Size([625]))

    def test_cnn_encoder_decoder(self):
        random_input = torch.rand((1, 3, 64, 64))
        random_output = torch.rand(8)

        encoder = DefaultCNNEncoder(input_channels=3, output_shape=8)
        decoder = DefaultCNNDecoder(output_channels=3, input_shape=8)

        assert(encoder(random_input).shape == torch.Size([8]))
        assert (decoder(random_output).shape == torch.Size([1,3,64,64]))
