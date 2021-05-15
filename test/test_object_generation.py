import unittest

from src.utils.model_types import ModelType
from src.utils.instance_provider import InstanceProvider


class ObjectGenerationTest(unittest.TestCase):

    def test_object_generation(self):
        path = "../config/cnn_ae.yaml"

        obj_gen = InstanceProvider(path, ModelType.CNN_AE)


if __name__ == "__main__":
    unittest.main()
