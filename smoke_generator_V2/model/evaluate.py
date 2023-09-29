import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from unet import UNET  # Import your UNet model
import matplotlib.pyplot as plt
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
# Load the trained model weights
model_weights_path = r'..\weight\best_model.pt'  # Update with the correct path
model_weights_path = os.path.join(script_dir,model_weights_path)
# Define the root directory where your evaluation images are located
eval_root_dir = r'..\real_images\images'
eval_root_dir = os.path.join(script_dir,eval_root_dir)
# Saving directory
saving_dir = os.path.join(eval_root_dir,"../masks")
if not(os.path.exists(saving_dir)):
  os.makedirs(saving_dir)
  print(f"{saving_dir} has been created ")

# Define the number of images you want to evaluate
num_images_to_evaluate = 512  # Change this to the number of images you want to evaluate

# Define data transforms for evaluation
eval_data_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Adjust the size as needed
    transforms.ToTensor(),
])

# Create a list of image filenames to load
eval_image_filenames = os.listdir(eval_root_dir)[:num_images_to_evaluate]

# Initialize the trained UNet model
in_channels = 3  # Number of input channels (e.g., RGB image)
out_channels = 1  # Number of output channels (e.g., binary segmentation)
model = UNET(in_channels, out_channels)

model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))  # Load model on CPU
model.eval()  # Set the model to evaluation mode

# Define a function to evaluate and visualize individual images
def evaluate_and_visualize(model, image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    image1 = eval_data_transform(image).unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        # Forward pass to generate predictions
        predicted_mask = model(image1)

        # Convert the predicted_mask to a format suitable for visualization
        predicted_mask = torch.sigmoid(predicted_mask[0]).squeeze().cpu().numpy()  # Assuming batch size is 1

        # Save the predicted mask as an image
        mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))
        mask_image = mask_image.resize(image.size)
        mask_image.save(os.path.join(saving_dir, os.path.basename(image_path)))
        """
        # Plot the original image and predicted mask
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(image)  # Convert from (C, H, W) to (H, W, C)
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(mask_image, cmap='gray')
        plt.title("Predicted Mask")

        plt.tight_layout()
        plt.show()
        """
        
# Call the evaluation function for each image
for image_filename in eval_image_filenames:
    image_path = os.path.join(eval_root_dir, image_filename)
    evaluate_and_visualize(model, image_path)
