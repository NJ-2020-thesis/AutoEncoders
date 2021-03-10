from typing import TypeVar

from pytorch_lightning import Trainer

from src.dataset_utils.vm_dataset import VisuomotorDataset
from src.transformation.transformation import CustomTransformation

# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')

import matplotlib

matplotlib.use('TkAgg', warn=False, force=True)

from src.autoencoders.vae_vanilla import VanillaVAE

import torch
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: VanillaVAE,
                 ) -> None:
        super(VAEXperiment, self).__init__()

        self.DATASET_PATH = "/home/anirudh/Desktop/main_dataset/**/*.png"
        self.batch_size = 32

        self.model = vae_model
        self.curr_device = None

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.batch_size/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        # self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.batch_size / self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          "{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          "recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        del test_input, recons #, samples


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
                                                             gamma = 0.95)
                scheds.append(scheduler)

                return optims, scheds
        except:
            return optims

    def train_dataloader(self):
        transform = CustomTransformation().get_transformation()
        train_dataset = VisuomotorDataset(self.DATASET_PATH, transform, (64, 64))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.num_train_imgs = len(train_loader)
        return train_loader

    def val_dataloader(self):
        transform = CustomTransformation().get_transformation()
        val_dataset = VisuomotorDataset(self.DATASET_PATH, transform, (64, 64))

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
        )
        self.num_val_imgs = len(val_loader)
        return val_loader


if __name__ == "__main__":
    input_dim = 28 * 28
    batch_size = 32

    model = VanillaVAE(in_channels=3,latent_dim=8)
    vaeX = VAEXperiment(model)

    runner = Trainer(gpus=1)
    runner.fit(vaeX)

