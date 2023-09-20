import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import logging
import matplotlib.pyplot as plt
# Import your UNet model and CustomDataset class
from unet import UNET
from custom_dataset import CustomDataset
import os

file_dir = os.path.abspath(__file__)
weight_dir = os.path.join(file_dir,"../../weight")

# Define hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 1
log_interval = 1
save_interval = 1  # Save model weights every epoch

# Define the root directory where your image and mask folders are located
root_dir = r'D:\mploi\Documents\Albatros\albatros\smoke_dataset_V1'

# Define data transforms if needed
data_transform = transforms.Compose([
    transforms.Resize((256,256)),  # Adjust the size as needed
    transforms.ToTensor(),
])

# Create the custom dataset
dataset = CustomDataset(root_dir, transform=data_transform)

# Split the dataset into training, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for the datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)  # Test DataLoader

# Initialize the UNet model
in_channels = 3  # Number of input channels (e.g., RGB image)
out_channels = 1  # Number of output channels (e.g., binary segmentation)
model = UNET(in_channels, out_channels)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize logging
logging.basicConfig(filename='training.log', level=logging.INFO)

# Training loop
best_val_loss = float('inf')  # Initialize with a high value
best_model_weights_path = None

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image']
        masks = batch['mask']
        
        optimizer.zero_grad()  # Zero the gradients
        
        # Forward pass
        outputs = model(images)
        
        # Calculate the loss
        loss = criterion(outputs, masks)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # Save model weights
    if (epoch + 1) % save_interval == 0:
        model_weights_path = os.path.join(weight_dir,f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_weights_path)
    
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    
    with torch.no_grad():
        for val_batch in val_loader:
            val_images = val_batch['image']
            val_masks = val_batch['mask']
            
            # Forward pass
            val_outputs = model(val_images)
            
            # Calculate the validation loss
            val_loss += criterion(val_outputs, val_masks).item()
    
    avg_val_loss = val_loss / len(val_loader)
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
    
    # Save the model if it has the best validation loss so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights_path = os.path.join(weight_dir,f"best_model.pt")
        torch.save(model.state_dict(), best_model_weights_path)

print("Training complete!")

# Load the best model for testing
test_model = UNET(in_channels, out_channels)
test_model.load_state_dict(torch.load(best_model_weights_path))
test_model.eval()  # Set the model to evaluation mode

# Testing loop on the test DataLoader
test_loss = 0.0
with torch.no_grad():
    for batch_idx, test_batch in enumerate(test_loader):
        test_images = test_batch['image']
        test_masks = test_batch['mask']

        # Forward pass
        test_outputs = test_model(test_images)
        test_loss += criterion(test_outputs, test_masks).item()
        # Visualize predictions for a few samples


avg_test_loss = test_loss / len(test_loader)
logging.info(f"Test Loss: {avg_test_loss:.4f}")
