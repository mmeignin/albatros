from PIL import Image
import os 
import matplotlib.pyplot as plt
import numpy as np

##-------------------------------------------------------------------------------------------------------------------------------------
##                         Script to perform EDA on background to determine best tresholds for unrealistic regions of fire appearnces
##-------------------------------------------------------------------------------------------------------------------------------------

def is_non_sky_pixel(pixel):
    """
    Determine if a pixel is a non-sky pixel based on color thresholds.

    Args:
        pixel (tuple): RGB values of the pixel.

    Returns:
        bool: True if the pixel is a non-sky pixel, False otherwise.
    """
    blue_threshold = 100
    whitish_threshold = 170
    pixel = np.int32(pixel)
    return not ((pixel[0] > whitish_threshold and pixel[1] > whitish_threshold and pixel[2] > whitish_threshold))

def create_sky_mask(image):
    """
    Create a binary mask to identify non-sky regions in an image.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: A binary mask where non-sky pixels are set to 255 and sky pixels to 0.
    """
    np_image = np.array(image)
    height, width, _ = np_image.shape

    # Create a boolean mask for non-sky pixels
    non_sky_mask = np.apply_along_axis(is_non_sky_pixel, 2, np_image)
    
    # Create the mask using the non_sky_mask
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[non_sky_mask] = 255

    return Image.fromarray(mask, "L")

def main():
    images_folder = r"..\background_images"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    background_folder = os.path.join(script_dir,images_folder)
    # Iterate through each image in the folder
    for filename in os.listdir(background_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") and "_4" in filename:
            original_image = Image.open(os.path.join(background_folder, filename))
            mask = create_sky_mask(original_image)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(mask, cmap="gray")
            axes[1].set_title("Sky Mask")
            axes[1].axis("off")

            plt.show()

if __name__ == "__main__":
    main()
