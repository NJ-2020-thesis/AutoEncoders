import matplotlib
import numpy as np
import torch

from src.autoencoders.model_factory import ModelFactory
from src.utils.model_types import ModelType

matplotlib.use('TkAgg', warn=False, force=True)


def get_image_representation(model_type: ModelType,
                             model_path: str,
                             img: np.array):
    model_factory = ModelFactory()
    model = model_factory.get_model(model_type)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output_img, representation = model(img)

    return output_img, representation


if __name__ == "__main__":
    ModelPath = ""
    resized_image = torch.rand((625))

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output, repr_vec = get_image_representation(ModelType.AE,
                                                ModelPath,
                                                resized_image)
    print(output.shape, repr_vec.shape)
