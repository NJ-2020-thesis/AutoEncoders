import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision

from src.autoencoders.cnn_autoencoder import ConvAutoencoder
from src.dataset_utils.vm_dataset import VisuomotorDataset
from src.transformation.transformation import CustomTransformation

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# --------------------------------------------------------------

seed = 42
torch.manual_seed(seed)

batch_size = 32
epochs = 300
learning_rate = 1e-3

DATASET_PATH = "/home/anirudh/Desktop/main_dataset/**/*.png"
MODEL_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model/" \
                  "cnn_ae_500_64.pth"
MODEL_SAVE_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model/" \
                  "cnn_ae_500_64.pth"
# --------------------------------------------------------------

transform = CustomTransformation().get_transformation()
train_dataset = VisuomotorDataset(DATASET_PATH,transform,(64,64))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = ConvAutoencoder().to(device)

# create an optimizer object
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# mean-squared error loss
criterion = nn.MSELoss()

for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:
        # reshape mini-batch data to [N,c,w,h] matrix
        # load it to the active device
        # batch_features = batch_features.view(-1, 784).to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(batch_features.cuda())

        # compute training reconstruction loss
        train_loss = criterion(outputs.cuda(), batch_features.cuda())

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

