import matplotlib
import torch
import torch.optim as optim

matplotlib.use('TkAgg',warn=False, force=True)

from src.autoencoders.spatial_autoencoder import DeepSpatialAutoencoder, DSAE_Loss
from src.dataset_utils.vm_dataset import VisuomotorDataset
from src.transformation.transformation import CustomTransformation

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# --------------------------------------------------------------
EPOCHS = 500
INPUT_SHAPE = (28, 28)
INPUT_DIMS = INPUT_SHAPE[0] * INPUT_SHAPE[1]
BATCH_SIZE = 512

DATASET_PATH = "/home/anirudh/Desktop/main_dataset/**/*.png"
MODEL_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model" \
                  "/cnn_ds_vae_small_1000_gpu.pth"
MODEL_SAVE_PATH = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/AutoEncoders/model" \
                  "/cnn_ds_vae_small_1000_gpu.pth"
# ---------------------------------------

transform = CustomTransformation().get_transformation()
train_dataset = VisuomotorDataset(DATASET_PATH, transform, INPUT_SHAPE)

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=2)

print('Number of samples: ', len(train_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds_vae = DeepSpatialAutoencoder(image_output_size=INPUT_SHAPE).to(device)

criterion = DSAE_Loss(False)

optimizer = optim.Adam(ds_vae.parameters(), lr=0.003)

for epoch in range(EPOCHS):
    for i, data in enumerate(dataloader, 0):
        inputs, classes = data
        inputs, classes = inputs.cuda(), classes.cuda()  # add this line

        optimizer.zero_grad()
        dec = ds_vae(inputs)
        loss,_ = criterion(dec, inputs)
        writer.add_scalar("Loss/train", loss, epoch)

        loss.backward()
        optimizer.step()
        l = loss.item()

    print(epoch, l)

torch.save(ds_vae.state_dict(), MODEL_PATH)

# plt.imshow(ds_vae(inputs.cuda()).data[5].numpy(), cmap='gray')
# plt.show(block=True)

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



