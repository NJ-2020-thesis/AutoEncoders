import matplotlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.utils.model_types import ModelType
from src.utils.name_list import *
from src.utils.vmp_dataset import VisuomotorDataset
from src.utils.instance_provider import InstanceProvider
matplotlib.use('TkAgg', warn=False, force=True)

seed = 42
torch.manual_seed(seed)


class AEXperiment(pl.LightningModule):
    """
    Pytorch Lightning module for training autoencoders.
    """
    def __init__(self,
                 config_path: str,
                 model_type: ModelType
                 ) -> None:
        super(AEXperiment, self).__init__()

        self.instance_provider = InstanceProvider(config_path, model_type)

        self.dataset_path = self.instance_provider.h_dataset_path
        self.model = self.instance_provider.model

        self.batch_size = self.instance_provider.h_batch_size
        self.epochs = self.instance_provider.h_epochs
        self.learning_rate = self.instance_provider.h_learning_rate
        self.dataset_img_size = self.instance_provider.h_img_size

        self.transform = self.instance_provider.transformation

        self.curr_device = None
        self.train_loader, self.val_loader = None, None
        self.num_train_imgs, self.num_val_imgs = None, None

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        self.curr_device = inputs.device

        outputs, _ = self.forward(inputs)
        train_loss = self.model.loss(inputs, outputs)

        return train_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        self.curr_device = inputs.device

        outputs, _ = self.forward(inputs)
        val_loss = self.model.loss(inputs, outputs)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        train_dataset = VisuomotorDataset(self.dataset_path,
                                          self.transform,
                                          self.dataset_img_size)

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=4)
        self.num_train_imgs = len(self.train_loader)
        return self.train_loader

    def val_dataloader(self):
        val_dataset = VisuomotorDataset(self.dataset_path,
                                        self.transform,
                                        self.dataset_img_size)

        self.val_loader = DataLoader(val_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=4)
        self.num_val_imgs = len(self.val_loader)
        return self.val_loader


if __name__ == "__main__":
    DATASET_PATH = "/home/anirudh/Desktop/main_dataset/**/*.png"

    model_trainer = AEXperiment(config_path="./config/cnn_ae.yaml",
                                model_type=ModelType.CNN_AE)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model_trainer,
                model_trainer.train_dataloader(),
                model_trainer.val_dataloader())
