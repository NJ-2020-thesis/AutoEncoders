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

input_dim = 28 * 28
batch_size = 512

DATASET_PATH = "/home/anirudh/Desktop/main_dataset/door_1/*.png"
MODEL_SAVE_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model/cnn_vae_test.pth"

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = VisuomotorDataset(DATASET_PATH,transform,(64,64))

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

print('Number of samples: ', len(train_dataset))

encoder = Encoder(input_dim, 100, 100)
decoder = Decoder(8, 100, input_dim)
vae = VAE(encoder, decoder)

criterion = nn.MSELoss()

optimizer = optim.Adam(vae.parameters(), lr=0.003)
l = None
for epoch in range(100):
    for i, data in enumerate(dataloader, 0):
        inputs, classes = data
        inputs, classes = Variable(inputs.resize_(batch_size, input_dim)), Variable(classes)
        optimizer.zero_grad()
        dec = vae(inputs)
        ll = latent_loss(vae.z_mean, vae.z_sigma)
        loss = criterion(dec, inputs) + ll
        loss.backward()
        optimizer.step()
        l = loss.item()
    print(epoch, l)

plt.imshow(vae(inputs).data[0].numpy().reshape(28, 28), cmap='gray')
plt.show(block=True)

torch.save(vae.state_dict(), MODEL_SAVE_PATH)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in vae.state_dict():
    print(param_tensor, "\t", vae.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])



