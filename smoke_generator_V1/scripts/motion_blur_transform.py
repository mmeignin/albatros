import cv2
import numpy as np
import os 
import matplotlib.pyplot as plt
from image_composer import apply_advanced_motion_blur
##-------------------------------------------------------------------------------------------------------------------
##                         Script to perform motion blur on images to determine best parameters
##-------------------------------------------------------------------------------------------------------------------

# input path
input_image_relative_path = "..\..\smoke_dataset_V1\images"
script_dir = os.path.dirname(os.path.abspath(__file__))
input_image_path = os.path.join(script_dir,input_image_relative_path)
# Motion Blur 
image_files = [f for f in os.listdir(input_image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

num_images_to_process = 5
# Create a subplot grid
num_rows = 2
num_cols = num_images_to_process
plt.figure(figsize=(15, 7))

for i in range(num_images_to_process):
    # Load the original image
    image_filename = image_files[i]
    image_path = os.path.join(input_image_path, image_filename)
    original_image = cv2.imread(image_path)

    # Apply motion blur
    motion_blur_image = apply_advanced_motion_blur(original_image, angle_degrees=90, blur_strength=15)

    # Plot the original image
    plt.subplot(num_rows, num_cols, i+1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # Plot the motion-blurred image
    plt.subplot(num_rows, num_cols, i+num_cols+1)
    plt.imshow(cv2.cvtColor(motion_blur_image, cv2.COLOR_BGR2RGB))
    plt.title("Motion Blurred Image")
    plt.axis('off')

plt.tight_layout()
plt.show()

