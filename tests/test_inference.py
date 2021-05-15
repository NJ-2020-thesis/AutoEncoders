import unittest

import torch

from ae_inference import get_image_representation
from src.utils.model_types import ModelType


class InferenceTest(unittest.TestCase):

    # @unittest.skip("Skipping CNN-AE inference tests!")
    def test_cnnae_inference(self):
        resized_image = torch.rand((10, 3, 64, 64)).cuda()
        output, repr_vec = get_image_representation(ModelType.CNN_AE,
                                                    "./../config/cnn_ae.yaml",
                                                    None,
                                                    resized_image)

        self.assertNotEqual(output.shape, torch.Size([10, 8]), 'CNN encoder error!')
        self.assertNotEqual(repr_vec.shape, torch.Size([10, 3, 64, 64]), 'CNN decoder error!')

    @unittest.skip("Skipping MLP-AE inference tests!")
    def test_mlp_ae_inference(self):
        resized_image = torch.rand((10, 625)).cuda()
        output_img, repr_vec = get_image_representation(ModelType.AE,
                                                        "./../config/mpl_ae.yaml",
                                                        None,
                                                        resized_image)

        self.assertNotEqual(output_img.shape, torch.Size([10, 625]), 'MLP encoder error!')
        self.assertNotEqual(repr_vec.shape, torch.Size([10, 8]), 'MLP decoder error!')


if __name__ == '__main__':
    unittest.main()
