import unittest

from src.utils.model_types import ModelType
from src.utils.instance_provider import ObjectGeneration


class ObjectGenerationTest(unittest.TestCase):

    def test_object_generation(self):
        path = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/config/cnn_ae.yaml"

        obj_gen = ObjectGeneration(path)

if __name__ == "__main__":
    unittest.main()
