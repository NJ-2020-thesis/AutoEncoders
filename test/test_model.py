import torch

import unittest
from src.autoencoders.autoencoder import AutoEncoder
from src.autoencoders.default_encoder_decoder import DefaultEncoder, DefaultDecoder, \
                                                        DefaultCNNEncoder, DefaultCNNDecoder


class ModelTest(unittest.TestCase):

    # def test_encoder_decoder(self):
    #     random_input = torch.rand(625)
    #     random_output = torch.rand(8)
    #
    #     encoder = DefaultEncoder(625, 8)
    #     decoder = DefaultDecoder(8, 625)
    #
    #     assert(encoder(random_input).shape == torch.Size([8]))
    #     assert(decoder(random_output).shape == torch.Size([625]))
    #
    # def test_ae(self):
    #     random_input = torch.rand(625)
    #     model = AutoEncoder(DefaultEncoder(625, 8), DefaultDecoder(8, 625))
    #     x, z = model(random_input)
    #
    #     assert(z.shape == torch.Size([8]))
    #     assert(x.shape == torch.Size([625]))

    def test_cnn(self):
        random_input = torch.rand((1,3,64,64))
        model = AutoEncoder(DefaultCNNEncoder(input_channels=3, output_shape=None),
                            DefaultCNNDecoder(input_channels=3, output_shape=None))
        x, z = model(random_input)

        print(x.shape)
        print(z.shape)
        # assert (z.shape == torch.Size([8]))
        # assert (x.shape == torch.Size([1, 3, 64, 64]))

if __name__ == '__main__':
    unittest.main()