import unittest

import torch

from src.utils.transform_utils import CustomTransformation


class TransformationTest(unittest.TestCase):

    def test_transformation(self):
        random_data = torch.rand((3, 256, 256))

        transformation = CustomTransformation()
        transformed_data = transformation.transform_data(random_data)
        self.assertEqual(transformed_data.shape, random_data.shape, "Transformation error!")


if __name__ == '__main__':
    unittest.main()
