# https://github.com/PyTorchLightning/pytorch-lightning/issues/525

import matplotlib
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from argparse import Namespace

matplotlib.use('TkAgg',warn=False, force=True)

from src.autoencoders.vae_autoencoder import VAE,Encoder,Decoder
from src.dataset_utils.vm_dataset import VisuomotorDataset
from src.autoencoders.vae_vanilla import VanillaVAE

# Hyper params
input_dim = 64 * 64
batch_size = 10

# Dataset
DATASET_PATH = "/home/anirudh/Desktop/main_dataset/**/*.png"
MODEL_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/src/training/lightning_logs/" \
             "version_0/checkpoints/epoch=33-step=6754.ckpt"

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = VisuomotorDataset(DATASET_PATH,transform,(64,64))

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

hparams = {
"batch_size":32,
"in_channels":3,
"latent_dim":8
}

vae = VanillaVAE.load_from_checkpoint(checkpoint_path=MODEL_PATH,
                                      hparams={"latent_dim":8})

with torch.no_grad():
    vae.sample(3,current_device="cuda:0")
