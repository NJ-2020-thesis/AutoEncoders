from pprint import pprint

import yaml

from src.autoencoders.model_factory import ModelFactory
from src.utils.model_types import ModelType
from src.utils.transform_utils import CustomTransformation


class InstanceProvider:
    """
    Dependency injection module for
    creating classes from a config file
    """

    def __init__(self, path: str, model_type: ModelType):
        with open(path, 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        pprint(data_loaded['model']['encoder'])

        # Setting training hyper-parameters
        self.h_dataset_path = data_loaded['trainer']['dataset_path']
        self.h_batch_size = int(data_loaded['trainer']['batch_size'])
        self.h_epochs = int(data_loaded['trainer']['epochs'])
        self.h_learning_rate = float(data_loaded['trainer']['optimizer']['learning_rate'])
        self.h_weight_decay = data_loaded['trainer']['optimizer']['weight_decay']
        self.h_img_size = (int(data_loaded['trainer']['img_size']['x']),
                           int(data_loaded['trainer']['img_size']['y']))
        self.h_gpu = int(data_loaded['trainer']['gpu'])

        # Dataset transformation
        self.transformation = CustomTransformation().get_transformation()

        # Generating the model
        model_factory = ModelFactory(data_loaded)
        self.model = model_factory.get_model(model_type)
