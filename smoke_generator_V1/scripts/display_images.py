import os
from PIL import Image
import random
import matplotlib.pyplot as plt

##-----------------------------------------------------------------------------------------
##                        Python Script to vizualise Smoke Simulations
##-----------------------------------------------------------------------------------------

def images_sample(main_folder_path, output_filename="synthetic_images_visualization.png", sample_size=5):
    # Calculate the number of rows and columns for subplots
    num_images = sample_size
    num_cols = 5  # Number of columns (images per row)
    num_rows = (num_images - 1) // num_cols + 1  # Calculate the number of rows

    # Create a subplot grid for displaying the composite images with reduced spacing
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows*3))
    
    # Adjust the horizontal spacing between subplots
    plt.subplots_adjust(wspace=0.02, hspace=0.05)  # Adjust wspace value as needed

    # Collect smoke image paths from subfolders
    images_path = []
    for files in os.listdir(main_folder_path):
        if files.endswith(".jpg") or files.endswith(".png"):  # Adjust for image file extensions
            images_path.append(os.path.join(main_folder_path, files))

    # Randomly select a sample of smoke images
    sample_images = random.sample(images_path, sample_size)

    # Composite each smoke image onto the background image and display the outputs
    for i, image_path in enumerate(sample_images):
        try:
            # Load the smoke image
            image = Image.open(image_path)
            # Calculate subplot position (row and column)
            row_idx = i // num_cols
            col_idx = i % num_cols

            # Display the composite image in the subplot
            ax = axs[row_idx, col_idx]
            ax.imshow(image)
            ax.axis('off')
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")

    # Remove empty subplots if there are fewer images than expected
    for i in range(sample_size, num_rows * num_cols):
        axs.flatten()[i].remove()

    # Save the composite image to a file
    plt.savefig(output_filename)
    plt.show()

# Example usage
main_folder_path = "../smoke_dataset_V1\images"  # Replace with your main folder path

# Call the smokes_sample function with your main folder path and customizations
images_sample(main_folder_path, output_filename="synthetic_images_visualization.png", sample_size=20)
