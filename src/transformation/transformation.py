# https://github.com/pytorch/vision/blob/master/examples/python/tensor_transforms.ipynb
# https://pytorch.org/docs/stable/torchvision/transforms.html

import torch
import torchvision
import torchvision.transforms as T


class CustomTransformation:
    def __init__(self):
        torch.manual_seed(12)
        self.transforms = T.RandomApply([
            T.RandomCrop(224),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomVerticalFlip(p=0.3),
            T.ColorJitter(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def transform_data(self, dataset):
        return self.transforms(dataset)


if __name__ == "__main__":
    print(torchvision.__version__)
