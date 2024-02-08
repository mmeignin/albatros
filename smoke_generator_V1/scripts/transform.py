import os
import cv2
from utils.transform_methods import apply_advanced_motion_blur, add_target_indicator, add_text_annotations, add_interface, add_monochromatic_noise, add_pixelisation, adjust_brightness, adjust_color_balance, apply_cutout

# Input and output folder paths
input_image_relative_path = "..\..\smoke_dataset_V1\images\\"
output_image_relative_path = "..\..\smoke_dataset_V1\output_images\\"  # Specify the output folder path

# Load the list of images
script_dir = os.path.dirname(os.path.abspath(__file__))
input_image_path = os.path.join(script_dir, input_image_relative_path)
output_image_path = os.path.join(script_dir, output_image_relative_path)
image_files = [f for f in os.listdir(input_image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Select the desired transformation function
transformation_function = apply_advanced_motion_blur  # Change this line to use a different transformation function

# Check if the output directory exists, otherwise create it
if not os.path.exists(output_image_path):
    os.makedirs(output_image_path)

# Loop through the images and apply the transformation
for image_filename in image_files:
    image_path = os.path.join(input_image_path, image_filename)
    original_image_cv2 = cv2.imread(image_path)

    # Apply the selected transformation to the image
    transformed_image_cv2 = transformation_function(original_image_cv2)

    # Save the transformed image to the output folder
    output_image_filename = "transformed_" + image_filename  # You can change the prefix as desired
    output_image_fullpath = os.path.join(output_image_path, output_image_filename)
    cv2.imwrite(output_image_fullpath, transformed_image_cv2)

print("Transformation applied to all images in the folder and saved in the output folder.")
