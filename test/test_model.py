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
        x_reconstruction, z = model(random_input)

        self.assertEqual(z.shape == torch.Size([8]), "AE model representation error!")
        self.assertEqual(x_reconstruction.shape == torch.Size([625]), "AE model reconstruction error!")

    def test_cnn_ae(self):
        random_input = torch.rand((1,3,64,64))
        model = AutoEncoder(DefaultCNNEncoder(input_channels=3, output_shape=8),
                            DefaultCNNDecoder(output_channels=3, input_shape=8))
        x_reconstruction, z = model(random_input)

        self.assertEqual(z.shape == torch.Size([8]), "CNN_AE model representation error!")
        self.assertEqual(x_reconstruction.shape == torch.Size([1, 3, 256, 256]), "CNN_AE model reconstruction error!")


if __name__ == '__main__':
    unittest.main()