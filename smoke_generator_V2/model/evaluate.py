import cv2
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from unet import UNET  # Import your UNet model
import matplotlib.pyplot as plt
import numpy as np
import random

# Define the root directory and file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_weights_path = os.path.join(script_dir, '..', 'weight', 'divine', 'best_model_iou.pt')
eval_root_dir = os.path.join(script_dir, '..', 'real_images', 'images')
eval_root_dir = r"smoke_dataset_V1\images"
saving_dir = os.path.join(script_dir, '..', 'real_images', 'masks')

# Ensure saving directory exists
#os.makedirs(saving_dir, exist_ok=True)
print(f"Saving directory: {saving_dir}")

# Define the number of images you want to evaluate
num_images_to_evaluate = 15  # Change this to the number of images you want to evaluate

# Define data transforms for evaluation
eval_data_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Adjust the size as needed
    transforms.ToTensor(),
])

# Load the trained UNet model
in_channels = 3  # Number of input channels (e.g., RGB image)
out_channels = 1  # Number of output channels (e.g., binary segmentation)
model = UNET(in_channels, out_channels)

model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))  # Load model on CPU
model.eval()  # Set the model to evaluation mode

def evaluate_and_display(model, image_path, saving_dir, threshold):
    try:
        # Load the image
        image = cv2.imread(image_path)

        # Convert the image to the appropriate format for the model
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_tensor = eval_data_transform(img_pil).unsqueeze(0)  # Add a batch dimension

        # Generate predictions using the UNet model
        with torch.no_grad():
            mask = model(img_tensor).squeeze().cpu().numpy()

        mask = (mask*255).astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Apply threshold to convert the mask values to binary (0 or 255)
        #print(np.unique(mask))
        tresholded_mask = (mask > threshold).astype(np.uint8) * 255

        # Resize the mask to match the original image dimensions
        tresholded_mask = cv2.resize(tresholded_mask, (image.shape[1], image.shape[0]))

        # Create a red filter
        red_filter = np.zeros_like(image)
        red_filter[tresholded_mask>0] = (0, 0, 255)  # Set the color to red

        # Apply the binary mask to the red filter
        red_filtered_image = cv2.addWeighted(image, 1, red_filter, 0.8, 0)

        #print(f"Unique values in the mask: {np.unique(mask)}")

        # Display the images using Matplotlib
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(wspace=0.02, hspace=0.05)  # Adjust wspace value as needed
        # Plot the original image
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title('Original Image')
        axs[0].axis('off')


        # Plot the image with red filter applied
        axs[1].imshow(cv2.cvtColor(red_filtered_image, cv2.COLOR_BGR2RGB))
        axs[1].set_title('Original Image with prediction overlay')
        axs[1].axis('off')

        # Plot the mask
        axs[2].imshow(mask, cmap='gray')
        axs[2].set_title('Predicted Mask')
        axs[2].axis('off')


        plt.show()

    except Exception as e:
        print(f"Error processing image {image_path}. Error: {e}")

# Evaluate and display each image in the evaluation directory
eval_image_filenames = os.listdir(eval_root_dir)
random.shuffle(eval_image_filenames)  # Randomize the list

# Select a subset of the randomized list
eval_image_subset = eval_image_filenames[:num_images_to_evaluate]

# Evaluate and display each image in the evaluation subset
for i, image_filename in enumerate(eval_image_subset):
    image_path = os.path.join(eval_root_dir, image_filename)
    print(f"Processing image {i + 1}/{len(eval_image_subset)}: {image_filename}")
    evaluate_and_display(model, image_path, saving_dir, threshold=50)
