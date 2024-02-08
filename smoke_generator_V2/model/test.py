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
import pickle

##-------------------------------------------------------------------------------------------------------------------
##                         Training Loop for Segmentation Models
##-------------------------------------------------------------------------------------------------------------------


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print("-----------------------------------------------------------------------")
print("Training Starting on device:" "gpu" if torch.cuda.is_available() else 'cpu')
print("-----------------------------------------------------------------------")

#-- training hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 200
log_interval = 1
save_interval = 25
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

# Define the root directory where test images and mask folders are located
root_dir = r''

#--Weight directory
file_dir = os.path.dirname(os.path.abspath(__file__))

# Define data transforms if needed
data_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Adjust the size as needed
    transforms.ToTensor(),
])

# Create the custom dataset
dataset = CustomDataset(root_dir, transform=data_transform)

## Data Split


# Create DataLoaders for the datasets
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Initialize the UNet model and move it to the GPU if available
in_channels = 3  # Number of input channels (e.g., RGB image)
out_channels = 1  # Number of output channels (e.g., binary segmentation)
model = UNET(in_channels, out_channels)
# Log the model architecture to wandb
#wandb.watch(model)

# Define loss function and optimizer
if criterion_name == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss
if optimizer_name == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Testing loop
# Load the trained model weights
model_weights_path = r'smoke_generator_V2\weight\noble-water-10\best_model_val_iou.pt'  # Update with the correct path
model.load_state_dict(torch.load(model_weights_path,map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

test_loss = 0.0
test_dice_scores = []
test_iou_scores = []

with torch.no_grad():
    for test_batch in test_loader:
        test_images = test_batch['image'].to(device)
        test_masks = test_batch['mask'].to(device)

        # Forward pass
        test_outputs = model(test_images)

        # Apply sigmoid activation and threshold to obtain binary predictions on the GPU
        test_outputs = (torch.sigmoid(test_outputs) > 0.5).float()
        test_masks = (test_masks > 0.5).float()
        #test_outputs = torch.ones_like(test_masks)
        #all_zeros_masks = torch.zeros_like(test_masks)


        # Calculate the test loss
        test_loss += criterion(test_outputs, test_masks).item()

        # Calculate Dice Coefficient (F1 Score) on the GPU
        dice = f1_score(test_masks.view(-1).cpu().numpy(), test_outputs.view(-1).cpu().numpy())
        test_dice_scores.append(dice)

        # Calculate Intersection over Union (IoU) on the GPU
        iou = jaccard_score(test_masks.view(-1).cpu().numpy(), test_outputs.view(-1).cpu().numpy())
        test_iou_scores.append(iou)

avg_test_loss = test_loss / len(test_loader)
avg_test_dice = sum(test_dice_scores) / len(test_dice_scores)
avg_test_iou = sum(test_iou_scores) / len(test_iou_scores)


print("-----------------------------------------------------------------------")
print("Testing Loop")
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Dice: {avg_test_dice:.4f}")
print(f"Test IoU: {avg_test_iou:.4f}")
print("-----------------------------------------------------------------------")

