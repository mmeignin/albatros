import os
import torch
import torchvision.transforms as transforms
from custom_dataset import CustomDataset  # Import your custom dataset class
from unet import UNET  # Import your UNet model
import matplotlib.pyplot as plt

# Define the root directory where your evaluation images are located
eval_root_dir = r'D:\mploi\Documents\Albatros\albatros\smoke_dataset_V1'

# Define the number of images you want to evaluate
num_images_to_evaluate = 10  # Change this to the number of images you want to evaluate
threshold =  0.5
# Define data transforms for evaluation
eval_data_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust the size as needed
    transforms.ToTensor(),
])

# Create a list of image filenames to load
eval_image_filenames = os.listdir(os.path.join(eval_root_dir, 'images'))[:num_images_to_evaluate]

# Create the evaluation dataset using the selected image filenames
eval_dataset = CustomDataset(eval_root_dir , transform=eval_data_transform)

# Create DataLoader for evaluation
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)  # Use batch size 1 for individual images

# Initialize the trained UNet model
in_channels = 3  # Number of input channels (e.g., RGB image)
out_channels = 1  # Number of output channels (e.g., binary segmentation)
model = UNET(in_channels, out_channels)

# Load the trained model weights
model_weights_path = r'D:\mploi\Documents\Albatros\albatros\smoke_generator_V2\weight\best_model.pt'  # Update with the correct path
model.load_state_dict(torch.load(model_weights_path))
model.eval()  # Set the model to evaluation mode

# Define a function to create masks and visualize the results
def evaluate_and_visualize(model, dataloader):
    with torch.no_grad():
        for batch in dataloader:
            image = batch['image']

            # Forward pass to generate predictions
            predicted_mask = model(image)
            
            
            predicted_mask = predicted_mask[0].squeeze().cpu().numpy()
            # Convert the predicted_mask to a format suitable for visualization
            # Plot the original image and predicted mask
            plt.figure(figsize=(8, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(image[0].permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            plt.title("Original Image")

            plt.subplot(1, 2, 2)
            plt.imshow(predicted_mask, cmap='gray')
            plt.title("Predicted Mask")

            plt.tight_layout()
            plt.show()

# Call the evaluation function
evaluate_and_visualize(model, eval_loader)
