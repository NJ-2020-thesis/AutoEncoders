import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch import nn
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg',warn=False, force=True)

from src.autoencoders.spatial_autoencoder import DeepSpatialAutoencoder, DSAE_Loss
from src.dataset_utils.vm_dataset import VisuomotorDataset

EPOCHS = 500
INPUT_SIZE = (64, 64)
INPUT_DIMS = INPUT_SIZE[0] * INPUT_SIZE[1]
BATCH_SIZE = 128

DATASET_PATH = "/home/anirudh/Desktop/main_dataset/**/*.png"
MODEL_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model" \
                  "/cnn_ds_vae_big_100_test.pth"

# ---------------------------------------

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = VisuomotorDataset(DATASET_PATH,transform,INPUT_SIZE)

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=2)

print('Number of samples: ', len(train_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds_vae = DeepSpatialAutoencoder(image_output_size=INPUT_SIZE).to(device)

criterion = DSAE_Loss(False)

optimizer = optim.Adam(ds_vae.parameters(), lr=0.0003)

for epoch in range(EPOCHS):
    for i, data in enumerate(dataloader, 0):
        inputs, classes = data
        optimizer.zero_grad()
        dec = ds_vae(inputs)
        loss,_ = criterion(dec, inputs)
        loss.backward()
        optimizer.step()
        l = loss.item()
    print(epoch, l)

torch.save(ds_vae.state_dict(), MODEL_PATH)

plt.imshow(ds_vae(inputs).data[5].numpy(), cmap='gray')
plt.show(block=True)

print("--------------------------------------")

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in ds_vae.state_dict():
    print(param_tensor, "\t", ds_vae.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

print("--------------------------------------")



