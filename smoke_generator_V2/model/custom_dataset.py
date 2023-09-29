import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, 'images')
        self.mask_folder = os.path.join(root_dir, 'masks')
        self.image_filenames = os.listdir(self.image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_folder, image_name)
        mask_path = os.path.join(self.mask_folder, image_name.replace(".jpg",".png").replace("image","mask"))  # Adjust mask file extension as needed
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return {'image': image, 'mask': mask}
"""
# Define data transforms if needed
data_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Adjust the size as needed
    transforms.ToTensor(),
])

root_dir= r'/content/drive/MyDrive/Data_Augmentation/smoke_generator_V2/smoke_dataset_V1'
# Create the custom dataset
dataset = CustomDataset(root_dir, transform=data_transform)

# Create a DataLoader for the dataset
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



for batch_idx, batch in enumerate(dataloader):
    images = batch['image']
    masks = batch['mask']

    
    for i in range(images.size(0)):
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title('Input Image')
        
        plt.subplot(1, 3, 2)
        plt.imshow(masks[i][0], cmap='gray')
        plt.title('Ground Truth Mask')
        

        
        plt.tight_layout()
        plt.show()
    break
"""