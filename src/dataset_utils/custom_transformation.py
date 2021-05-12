import torch
import torchvision
import torchvision.transforms as T


class CustomTransformation:
    def __init__(self):
        torch.manual_seed(12)
        self.transforms = T.Compose([
            # T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomVerticalFlip(p=0.3),
            T.ColorJitter(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def transform_data(self, dataset):
        return self.transforms(dataset)

    def get_transformation(self):
        return self.transforms