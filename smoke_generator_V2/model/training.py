import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
import os
from unet import UNET
from custom_dataset import CustomDataset
from sklearn.metrics import f1_score, jaccard_score

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
##-------------------------------------------------------------------------------------------------------------------
##                         Training Loop for Segmentation Models
##-------------------------------------------------------------------------------------------------------------------

#--Weight directory
file_dir = os.path.dirname(os.path.abspath(__file__))
weight_dir = os.path.join(file_dir, "../../weight")
if not os.path.exists(weight_dir):
    # If it doesn't exist, create it
    os.makedirs(weight_dir)
    print(f"{weight_dir} has been created")
else :
  print(f"model weights can be found at: {weight_dir}")
#-- training hyperparameters
batch_size = 4
learning_rate = 0.001
num_epochs = 100
log_interval = 1
save_interval = 1
criterion_name = 'BCEWithLogitsLoss'
optimizer_name = 'Adam'
##config for wandb logging
config = {
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'num_epochs': num_epochs,
    'criterion': criterion_name,  # Add the name of the criterion here
    'optimizer': optimizer_name,  # Add the name of the optimizer here
}

wandb.login(key='f2835223dda80f88689f728711a4bce69f85b8e7')
# Initialize wandb run with the updated config
wandb.init(project='albatros_smoke_segmentation', config=config)

## Data Loading

# Define the root directory where your image and mask folders are located
root_dir = r'/content/drive/MyDrive/Data_Augmentation/smoke_generator_V2/smoke_dataset_V1'

# Define data transforms if needed
data_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Adjust the size as needed
    transforms.ToTensor(),
])

# Create the custom dataset
dataset = CustomDataset(root_dir, transform=data_transform)

## Data Split
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for the datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)  # Test DataLoader

# Initialize the UNet model and move it to the GPU if available
in_channels = 3  # Number of input channels (e.g., RGB image)
out_channels = 1  # Number of output channels (e.g., binary segmentation)
model = UNET(in_channels, out_channels).to(device)

# Log the model architecture to wandb
wandb.watch(model)

# Define loss function and optimizer
if criterion_name == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss
if optimizer_name == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
best_val_loss = float('inf')  # Initialize with a high value
best_model_weights_path = None

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, masks)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    wandb.log({"Train Loss": avg_train_loss},step=epoch)
    print(f'Training Loop of Epoch:{epoch+1}/{num_epochs} over average training loss : {avg_train_loss}')
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for val_batch in val_loader:
          val_images = val_batch['image'].to(device)
          val_masks = val_batch['mask'].to(device)

          # Forward pass
          val_outputs = model(val_images)

          # Apply sigmoid activation and threshold to obtain binary predictions on the GPU
          val_outputs = (torch.sigmoid(val_outputs) > 0.5).float()
          val_masks = (val_masks > 0.5).float()

          # Calculate the validation loss
          val_loss += criterion(val_outputs, val_masks).item()

          # Calculate Dice Coefficient (F1 Score) on the GPU
          dice = f1_score(val_masks.view(-1).cpu().numpy(), val_outputs.view(-1).cpu().numpy())
          dice_scores.append(dice)

          # Calculate Intersection over Union (IoU) on the GPU
          iou = jaccard_score(val_masks.view(-1).cpu().numpy(), val_outputs.view(-1).cpu().numpy())
          iou_scores.append(iou)

    avg_val_loss = val_loss / len(val_loader)
    avg_dice_score = sum(dice_scores) / len(dice_scores)
    avg_iou_score = sum(iou_scores) / len(iou_scores)

    # Log metrics to wandb
    wandb.log({"Validation Loss": avg_val_loss, "Dice Coefficient (F1 Score)": avg_dice_score, "IoU": avg_iou_score},step=epoch)

    # Log images for one sample
    sample_images = val_images[:1]  # Take the first image from the validation set
    sample_masks = val_masks[:1]
    sample_outputs = val_outputs[:1]

    # Convert tensor images to numpy arrays
    sample_images = sample_images.permute(0, 2, 3, 1).cpu().numpy()
    sample_masks = sample_masks.permute(0, 2, 3, 1).cpu().numpy()
    sample_outputs = (sample_outputs > 0.5).permute(0, 2, 3, 1).cpu().numpy()
    
    # Log images to wandb
    wandb.log({
        "Sample Image": [wandb.Image(np.uint8(sample_images[0] * 255))],
        "Sample Mask": [wandb.Image(np.uint8(sample_masks[0] * 255))],
        "Sample Output": [wandb.Image(np.uint8(sample_outputs[0] * 255))]
    },step=epoch)

    # Print to terminal
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Dice: {avg_dice_score:.4f}, IoU: {avg_iou_score:.4f}")

    # Save model weights
    if (epoch + 1) % save_interval == 0:
        model_weights_path = os.path.join(weight_dir, f"model_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), model_weights_path)

    # Save the model if it has the best validation loss so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights_path = os.path.join(weight_dir, "best_model.pt")
        torch.save(model.state_dict(), best_model_weights_path)

print("Training complete!")
