import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch import nn

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg',warn=False, force=True)

from src.autoencoders.vae_autoencoder import VAE,Encoder,Decoder,latent_loss
from src.dataset_utils.vm_dataset import VisuomotorDataset

input_dim = 64 * 64
batch_size = 10

DATASET_PATH = "/home/anirudh/Desktop/main_dataset/**/*.png"
MODEL_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model" \
             "/cnn_vae_test_1.pth"

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = VisuomotorDataset(DATASET_PATH,transform,(64,64))

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

encoder = Encoder(input_dim, 100, 100)
decoder = Decoder(8, 100, input_dim)
vae = VAE(encoder, decoder)
vae.load_state_dict(torch.load(MODEL_PATH))
vae.eval()

with torch.no_grad():
    for i, data in enumerate(dataloader, 0):
        inputs, classes = data
        inputs, classes = inputs.resize_(batch_size, input_dim),classes

print(inputs.shape)

with torch.no_grad():
    forward1 = vae(inputs[8])
    forward2 = vae(inputs[3])

    print(forward1 - forward2)
    # plt.imshow(inputs[2].reshape(64, 64), cmap='gray')

    plt.imshow(forward1.numpy().reshape(64, 64), cmap='gray')
    plt.show(block=True)

