import matplotlib
import numpy as np
import torch

from src.ae_training import AEXperiment
from src.utils.model_types import ModelType


def get_image_representation(model_type: ModelType,
                             config_path: str,
                             model_path: str,
                             img: np.array):
    ae_inference = AEXperiment(config_path, model_type)
    ae_inference.load_state_dict(torch.load(model_path))
    ae_inference.eval()

    with torch.no_grad():
        output_img, representation = ae_inference(img)

    return output_img, representation
