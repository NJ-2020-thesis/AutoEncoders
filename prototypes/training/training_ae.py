# https://gist.github.com/AFAgarap/4f8a8d8edf352271fa06d85ba0361f26

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from src.autoencoders.basic_autoencoder import AutoEncoder
from src.dataset_utils.vm_dataset import VisuomotorDataset
from src.transformation.transformation import CustomTransformation

writer = SummaryWriter()

# --------------------------------------------------------------

seed = 42
torch.manual_seed(seed)

batch_size = 512
epochs = 50
learning_rate = 1e-4

DATASET_PATH = "/home/anirudh/Desktop/main_dataset/**/*.png"
MODEL_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model/" \
                  "gpu_ae_prototype.pth"
MODEL_SAVE_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model/" \
                  "ae_10_200.pth"
# --------------------------------------------------------------

transform = CustomTransformation().get_transformation()
train_dataset = VisuomotorDataset(DATASET_PATH,transform,(64,64))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AutoEncoder(input_shape=64*64,output_shape=10).to(device)
model.load_state_dict(torch.load(MODEL_PATH))

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# mean-squared error loss
criterion = nn.MSELoss()

for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:
        # load it to the active device
        # batch_features = batch_features.view(-1, 784).to(device)
        batch_features = batch_features.view(-1, 64*64).to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(batch_features)

        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)

        writer.add_scalar("Loss/train", train_loss, epoch)

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

torch.save(model.state_dict(), MODEL_SAVE_PATH)

# --------------------------------------------------------------

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
# --------------------------------------------------------------

writer.flush()