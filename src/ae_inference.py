from typing import Tuple
import numpy as np
import torch
import cv2

from src.ae_training import AEXperiment
from src.utils.model_types import ModelType


def get_image_representation(model_type: ModelType,
                             config_path: str,
                             model_path: str,
                             img: np.array) -> Tuple[np.array, np.array]:
    """
    Function to extract representation vector from test_image
    :param model_type: Autoencoder architecture types
    :param config_path: Path to AE architecture config
    :param model_path: Path to trained model
    :param img: Image input
    :return: Return the reconstructed test_image along with the test_image representation vector
    """
    ae_inference = AEXperiment(config_path, model_type)

    # Skips during testing
    if model_path:
        ae_inference.load_state_dict(torch.load(model_path))

    ae_inference.eval()

    with torch.no_grad():
        output_img, representation = ae_inference(img)

    return output_img, representation


if __name__ == "__main__":

    img = cv2.imread("")

    MODEL_PATH = ""
    output_img, repr_vector = get_image_representation(model_type=ModelType.CNN_AE,
                                                       config_path="./config/cnn_ae.yaml",
                                                       model_path=MODEL_PATH,
                                                       img=img)
