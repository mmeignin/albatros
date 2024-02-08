import os
from PIL import Image
import random
import matplotlib.pyplot as plt

##-----------------------------------------------------------------------------------------
##                        Python Script to vizualise Smoke Simulations
##-----------------------------------------------------------------------------------------
import os
from PIL import Image
import random
import matplotlib.pyplot as plt

def smokes_sample(main_folder_path, output_filename="smoke_simulation_visualization.png", sample_size=5):
    # Calculate the number of rows and columns for subplots
    num_images = sample_size
    num_cols = 5  # Number of columns (images per row)
    num_rows = (num_images - 1) // num_cols + 1  # Calculate the number of rows

    # Create a subplot grid for displaying the composite images with reduced spacing
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows*3))
    
    # Adjust the horizontal spacing between subplots
    plt.subplots_adjust(wspace=0.02, hspace=0.05)  # Adjust wspace value as needed

    # Create the background image with a softer black color
    n, m = (1024, 1024)
    background_color = (0, 0, 0, 0)  # RGBA tuple with reduced alpha value
    background = Image.new('RGBA', (n, m), background_color)

    # Collect smoke image paths from subfolders
    smoke_paths = []
    for root, dirs, files in os.walk(main_folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):  # Adjust for image file extensions
                smoke_paths.append(os.path.join(root, file))

    # Randomly select a sample of smoke images
    sample_images = random.sample(smoke_paths, sample_size)

    # Composite each smoke image onto the background image and display the outputs
    for i, smoke_path in enumerate(sample_images):
        try:
            # Load the smoke image
            smoke = Image.open(smoke_path)

            # Resize the smoke image to match the background image's dimensions
            smoke = smoke.resize(background.size)

            # Composite the smoke image onto the background image
            composite = Image.alpha_composite(background, smoke)

            # Calculate subplot position (row and column)
            row_idx = i // num_cols
            col_idx = i % num_cols

            # Display the composite image in the subplot
            ax = axs[row_idx, col_idx]
            ax.imshow(composite)
            ax.axis('off')
        except Exception as e:
            print(f"Error processing image {smoke_path}: {str(e)}")

    # Remove empty subplots if there are fewer images than expected
    for i in range(sample_size, num_rows * num_cols):
        axs.flatten()[i].remove()

    # Save the composite image to a file
    #plt.savefig(output_filename)
    plt.show()

# Example usage
main_folder_path = "../blender_images"  # Replace with your main folder path

# Call the smokes_sample function with your main folder path and customizations
smokes_sample(main_folder_path, output_filename="smoke_simulation_visualization.png", sample_size=10)
