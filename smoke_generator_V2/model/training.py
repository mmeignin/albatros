import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from unet import UNet  # Import your UNet model
from custom_dataset import CustomDataset  # Import your CustomDataset class

# Define hyperparameters
batch_size = 4
learning_rate = 0.001
num_epochs = 10

# Define the root directory where your image and mask folders are located
root_dir = r'D:\mploi\Documents\Albatros\albatros\smoke_dataset_V1'

# Define data transforms if needed
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust the size as needed
    transforms.ToTensor(),
])

# Create the custom dataset
dataset = CustomDataset(root_dir, transform=data_transform)

# Create a DataLoader for the dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the UNet model
in_channels = 3  # Number of input channels (e.g., RGB image)
out_channels = 1  # Number of output channels (e.g., binary segmentation)
model = UNet(in_channels, out_channels)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    
    for batch_idx, batch in enumerate(dataloader):
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
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

print("Training complete!")
