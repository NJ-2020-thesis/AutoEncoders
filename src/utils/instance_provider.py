import yaml
from pprint import pprint

from src.autoencoders.model_factory import ModelFactory
from src.utils.transform_utils import CustomTransformation
from src.utils.model_types import ModelType

class ObjectGeneration:
    """
    Dependency injection module
    """

    def __init__(self, path: str):

        with open(path, 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        pprint(data_loaded['model']['encoder'])

        self.h_dataset_path = data_loaded['trainer']['dataset_path']
        self.h_batch_size = data_loaded['trainer']['batch_size']
        self.h_epochs = data_loaded['trainer']['epochs']
        self.h_learning_rate = data_loaded['trainer']['optimizer']['learning_rate']
        self.h_dataset_img_size = data_loaded['trainer']['optimizer']['weight_decay']

