import os
import random
import pandas as pd
from PIL import Image
from utils.image_composition import get_bounding_boxes,create_binary_mask # Import your get_bounding_boxes function
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage import io, color
import numpy as np
from skimage.filters import threshold_mean,threshold_local,threshold_yen,threshold_li,threshold_triangle

##-------------------------------------------------------------------------------------------------------------------
##                         Script to perform EDA on smoke bounding boxes to determine best treshold
##-------------------------------------------------------------------------------------------------------------------

def determine_optimal_threshold(image_paths):
    """
    Determine the optimal threshold value for a list of smoke image paths.

    Args:
        image_paths (list): List of paths to smoke images.

    Returns:
        int: Optimal threshold value.
    """
    optimal_threshold = 0

    for image_path in image_paths:
        # Load the image and convert to grayscale
        image = io.imread(image_path)
                
        # Remove the alpha channel if present

        if image.shape[2] == 4:  # Check if the image has an alpha channel
            image = image[:, :, :3]  # Keep only the RGB channels
        gray_image = color.rgb2gray(image)
        # Determine the optimal threshold using Otsu's method
        threshold_value = threshold_otsu(gray_image)
        print(threshold_value)
        # Accumulate the threshold values for later averaging
        optimal_threshold += threshold_value

    # Calculate the average threshold value for the folder
    average_threshold = optimal_threshold / len(image_paths)

    return int(average_threshold)

def display_images_with_otsu_threshold(images_folder):
    """
    Determine the optimal threshold value for a list of smoke image paths.

    Args:
        images_folder (list): folder containing to smoke images.

    Returns:
        int: Optimal threshold value.
    """
    image_files = os.listdir(images_folder)
    image_files = [f for f in image_files if f.endswith('.png')]

    num_images = min(4, len(image_files))

    fig, axes = plt.subplots(3, num_images, figsize=(15, 6))

    for i in range(num_images):
        image_path = os.path.join(images_folder, image_files[i])

        image = io.imread(image_path)
        alpha_channel = image[:, :, 3]  # Extract alpha channel

        threshold_value = threshold_triangle(alpha_channel)
        binary_mask = alpha_channel > threshold_value

        alpha_threshold_value = threshold_otsu(alpha_channel) 
        alpha_binary_mask = alpha_channel >= alpha_threshold_value


        axes[0, i].imshow(image)
        axes[0, i].set_title('Original Image')
        axes[0, i].axis('off')

        axes[1, i].imshow(alpha_binary_mask, cmap='gray')
        axes[1, i].set_title(f'alpha Mask (Otsu){alpha_threshold_value:.2f}')
        axes[1, i].axis('off')

        axes[2, i].imshow(binary_mask, cmap='gray')
        axes[2, i].set_title(f'Adverse method Mask')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

smoke_folder = r'..\blender_images\smokeplume_8'
script_dir = os.path.dirname(os.path.abspath(__file__))
smoke_folder = os.path.join(script_dir,smoke_folder)
image_paths = [smoke_folder + "\\" + x  for x in os.listdir(smoke_folder)]
display_images_with_otsu_threshold(smoke_folder)

