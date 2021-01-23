import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch import nn

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg',warn=False, force=True)

import sys
sys.path.insert(1, '/home/anaras2s/NJ-2020-thesis/AutoEncoders')

from src.autoencoders.vae_autoencoder import VAE,Encoder,Decoder
from src.dataset_utils.vm_dataset import VisuomotorDataset

# --------------------------------------------------------------

EPOCHS = 100
INPUT_SIZE = (64,64)
INPUT_DIMS = INPUT_SIZE[0] * INPUT_SIZE[1]
BATCH_SIZE = 512

DATASET_PATH = "/home/anaras2s/anirudh/main_dataset/**/*.png"
MODEL_PATH = "/home/model/VAE/" \
             "cnn_vae_test_1000_gpu.pth"
# --------------------------------------------------------------

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = VisuomotorDataset(DATASET_PATH,transform,INPUT_SIZE)

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=4)

print('Number of samples: ', len(train_dataset))

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(INPUT_DIMS, 100, 100)
decoder = Decoder(16, 100, INPUT_DIMS)
vae = VAE(encoder, decoder).to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(vae.parameters(), lr=0.0003)

l = None
for epoch in range(EPOCHS):
    for i, data in enumerate(dataloader, 0):
        inputs, classes = data
        inputs, classes = inputs.cuda(), classes.cuda()  # add this line

        inputs, classes = Variable(inputs.resize_(BATCH_SIZE, INPUT_DIMS)), \
                          Variable(classes)
        optimizer.zero_grad()
        dec = vae(inputs)
        ll = vae.latent_loss(vae.z_mean, vae.z_sigma)
        loss = criterion(dec, inputs) + ll
        loss.backward()
        optimizer.step()
        l = loss.item()
    print(epoch, l)

torch.save(vae.state_dict(), MODEL_PATH)

plt.imshow(vae(inputs.cuda()).data[5].numpy().reshape(INPUT_SIZE), cmap='gray')
plt.show(block=True)

print("--------------------------------------")
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in vae.state_dict():
    print(param_tensor, "\t", vae.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

print("--------------------------------------")

