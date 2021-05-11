import torch

import unittest
from src.autoencoders.autoencoder import AutoEncoder
from src.autoencoders.layer_utils import DefaultEncoder, DefaultDecoder, \
                                                        DefaultCNNEncoder, DefaultCNNDecoder


class ModelTest(unittest.TestCase):

    def test_ae(self):
        random_input = torch.rand(625)
        model = AutoEncoder(DefaultEncoder(input_shape=625, output_shape=8),
                            DefaultDecoder(input_shape=8, output_shape=625))
        x, z = model(random_input)

        assert(z.shape == torch.Size([8]))
        assert(x.shape == torch.Size([625]))

    def test_cnn_ae(self):
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