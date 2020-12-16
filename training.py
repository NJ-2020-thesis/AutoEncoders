# https://gist.github.com/AFAgarap/4f8a8d8edf352271fa06d85ba0361f26

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision

from src.autoencoders.basic_autoencoder import AutoEncoder
from src.dataset.vm_dataset import VisuomotorDataset

seed = 42
torch.manual_seed(seed)

batch_size = 512
epochs = 5
learning_rate = 1e-3

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# train_dataset = torchvision.datasets.MNIST(
#     root="~/torch_datasets", train=True, transform=transform, download=True
# )
DATASET_PATH = "/home/anirudh/Desktop/main_dataset/door_1/*.png"
train_dataset = VisuomotorDataset(DATASET_PATH,transform,(28,28))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AutoEncoder(input_shape=784,output_shape=16).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# mean-squared error loss
criterion = nn.MSELoss()

for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        # batch_features = batch_features.view(-1, 784).to(device)
        batch_features = batch_features.view(-1, 784).to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(batch_features)

        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_loader)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
