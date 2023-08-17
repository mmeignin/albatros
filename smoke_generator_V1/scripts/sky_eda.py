from PIL import Image
import os 
import matplotlib.pyplot as plt
import numpy as np

def is_non_sky_pixel(pixel):
    blue_threshold = 100
    whitish_threshold = 170
    pixel = np.int32(pixel)
    return not ((pixel[2] > blue_threshold and pixel[2] - pixel[1] > 15 and pixel[2] - pixel[0] > 15) or
                (pixel[0] > whitish_threshold and pixel[1] > whitish_threshold and pixel[2] > whitish_threshold))

def create_sky_mask(image):
    np_image = np.array(image)
    height, width, _ = np_image.shape

    # Create a boolean mask for non-sky pixels
    non_sky_mask = np.apply_along_axis(is_non_sky_pixel, 2, np_image)
    
    # Create the mask using the non_sky_mask
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[non_sky_mask] = 255

    return Image.fromarray(mask, "L")

def main():
    background_folder = r"D:\mploi\Documents\Albatros\albatros\smoke_generator_V1\background_images"

    # Iterate through each image in the folder
    for filename in os.listdir(background_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
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
