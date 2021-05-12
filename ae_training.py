import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from src.autoencoders.model_factory import ModelFactory
from src.dataset_utils.custom_transformation import CustomTransformation
from src.dataset_utils.model_types import ModelType
from src.dataset_utils.vmp_dataset import VisuomotorDataset
from src.dataset_utils.name_list import *

matplotlib.use('TkAgg', warn=False, force=True)

seed = 42
torch.manual_seed(seed)

# batch_size = 512
# epochs = 50
# learning_rate = 1e-4
# dataset_size = (64, 64)

writer = SummaryWriter()

import torch
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils


class AEXperiment(pl.LightningModule):

    def __init__(self,
                 model_type: ModelType,
                 dataset_path: str,
                 batch_size: int = 512,
                 epochs: int = 50,
                 learning_rate: float = 1e-4,
                 dataset_img_size: tuple = (64,64)) -> None:

        super(AEXperiment, self).__init__()

        self.model_factory = ModelFactory()
        self.dataset_path = dataset_path
        self.model = self.model_factory.get_model(model_type)

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dataset_img_size = dataset_img_size

        self.transform = CustomTransformation().get_transformation()

        self.curr_device = None
        self.train_loader, self.val_loader = None, None
        self.num_train_imgs, self.num_val_imgs = None, None

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss(real_img, results)

        # self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        # val_loss = self.model.loss_function(*results,
        #                                     M_N=self.batch_size / self.num_val_imgs,
        #                                     optimizer_idx=optimizer_idx,
        #                                     batch_idx=batch_idx)
        val_loss = self.model.loss(real_img, results)

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=1e-4,
                               weight_decay=0.0)
        optims.append(optimizer)

        try:
            if True:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=0.95)
                scheds.append(scheduler)

                return optims, scheds
        except:
            return optims

    def train_dataloader(self):
        train_dataset = VisuomotorDataset(self.dataset_path,
                                          self.transform,
                                          self.dataset_img_size)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.num_train_imgs = len(self.train_loader)
        return self.train_loader

    def val_dataloader(self):
        val_dataset = VisuomotorDataset(self.dataset_path,
                                        self.transform,
                                        self.dataset_img_size)

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.num_val_imgs = len(self.val_loader)
        return self.val_loader


if __name__ == "__main__":
    DATASET_PATH = "/home/anirudh/Desktop/main_dataset/**/*.png"

    model_trainer = AEXperiment(model_type=ModelType.CNN_AE,
                                dataset_path=DATASET_PATH,
                                dataset_img_size=(64, 64))
    trainer = pl.Trainer()
    trainer.fit(model_trainer,
                model_trainer.train_dataloader(),
                model_trainer.val_dataloader())