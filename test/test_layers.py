import torch
from torchsummary import summary

import unittest
from src.autoencoders.autoencoder import AutoEncoder
from src.autoencoders.layer_utils import DefaultEncoder, DefaultDecoder, \
                                                        DefaultCNNEncoder, DefaultCNNDecoder


class LayerTest(unittest.TestCase):

    # @unittest.skip("Skipping MPL Layer Test")
    def test_mlp_encoder_decoder(self):
        random_input = torch.rand(625) #25x25
        random_output = torch.rand(8)

        encoder = DefaultEncoder(input_shape=625, output_shape=8)
        decoder = DefaultDecoder(input_shape=8, output_shape=625)

        self.assertEqual(encoder(random_input).shape, torch.Size([8]), 'MPL encoder error!')
        self.assertEqual(decoder(random_output).shape, torch.Size([625]), 'MPL decoder error!')

    # @unittest.skip("Skipping CNN Layer Tests")
    def test_cnn_encoder_decoder(self):
        random_input = torch.rand((1, 3, 64, 64))
        random_output = torch.rand(8)

        encoder = DefaultCNNEncoder(input_channels=3, output_shape=8)
        decoder = DefaultCNNDecoder(output_channels=3, input_shape=8)

        self.assertEqual(encoder(random_input).shape, torch.Size([8]), 'CNN encoder error!')
        # self.assertEqual (decoder(random_output).shape, torch.Size([1,3,256,256]), 'CNN decoder error!')
