import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.transform_methods import apply_advanced_motion_blur, add_target_indicator, add_text_annotations,add_interface,add_monochromatic_noise,add_pixelisation,adjust_brightness,adjust_color_balance,apply_cutout
from PIL import Image

##-------------------------------------------------------------------------------------------------------------------
##                         Script to Show different type of transformations
##-------------------------------------------------------------------------------------------------------------------

# Define a list of transformation functions with the desired order
transformations = [
    ("No Transformation", lambda img: img),       # No transformations at all
    ("Motion Blur", apply_advanced_motion_blur),  # Apply motion blur alone
    #("Target Indicator", add_target_indicator),    # Apply target indicator alone
    #("UI indications", add_text_annotations),    # apply text annotation alone
    ("interface", add_interface),    # Apply target indicator alone
    ("Motion Blur + Target Indicator + text_annotations", lambda img: add_text_annotations(add_target_indicator(apply_advanced_motion_blur(img)))),  # Apply motion blur and target indicator together
    ("Motion Blur + interface",lambda img : add_interface(apply_advanced_motion_blur(img))), # Apply motion blur and an interface 
    ("Grain",add_monochromatic_noise),
    ("Pixelisation",add_pixelisation),
    ("Brightness",adjust_brightness),
    ("Color Balance",adjust_color_balance),
    ("Apply Cutout",apply_cutout)
]

if __name__ == "__main__":
    input_image_relative_path = "..\..\smoke_dataset_V1\images\\"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_image_path = os.path.join(script_dir, input_image_relative_path)

    image_files = [f for f in os.listdir(input_image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    num_images_to_process = 10 # Number of compositions to create and display
    num_cols =  5 # Number of columns in the grid
    num_rows = (num_images_to_process + num_cols - 1) // num_cols  # Calculate the number of rows

    plt.figure(figsize=(15, 7))

    for i in range(num_images_to_process):
        # Load the original image
        image_filename = random.choice(image_files)
        image_path = os.path.join(input_image_path, image_filename)
        original_image_cv2 = cv2.imread(image_path)

        # Apply a specific composition of transformations based on the desired order
        title, transform_function = transformations[i % len(transformations)]
        composed_image_cv2 = transform_function(original_image_cv2)
        
        cv2.imwrite(os.path.join(script_dir,"test_transform_2.png"),composed_image_cv2)
        composed_image_cv2 = cv2.cvtColor(composed_image_cv2, cv2.COLOR_BGR2RGB)
        # Plot the composed image with title
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(composed_image_cv2)
        #plt.tight_layout()
        plt.title(title)
        plt.axis('off')
        #plt.show()
    
    plt.tight_layout()
    plt.savefig(r"D:\mploi\Documents\Albatros\albatros\smoke_generator_V1\scripts\test_transform_2.png")
    plt.show()
