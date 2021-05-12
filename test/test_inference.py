import torch

import unittest

from src.dataset_utils.model_types import ModelType
from ae_inference import get_image_representation


class InferenceTest(unittest.TestCase):

    def test_inference(self):
        resized_image = torch.rand(625)
        output, repr_vec = get_image_representation(ModelType.AE,
                                                    "",
                                                    resized_image)

        self.assertNotEqual(output.shape, torch.Size([8]), 'MPL encoder error!')
        self.assertNotEqual(repr_vec.shape, torch.Size([625]), 'MPL decoder error!')


if __name__ == '__main__':
    unittest.main()
