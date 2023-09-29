from PIL import Image,ImageEnhance,ImageFilter
import os
import random
import numpy as np
from skimage.filters import threshold_otsu
import cv2

##-----------------------------------------------------------------------------------------
##                        methods for Image Composition
##-----------------------------------------------------------------------------------------
def get_bounding_boxes(smoke_image, threshold=50):
    """
    Extracts the bounding boxes of the smoke from the given smoke image.

    Args:
        smoke_image (Image): The smoke image as a PIL Image object.
        threshold (int, optional): The threshold value for smoke detection. Default is 50.

    Returns:
        Image: A PIL Image containing only the bounding boxes of the smoke, or None if no smoke is found.
    """
    # Convert the smoke image to grayscale
    grayscale_image = smoke_image.convert("L")

    # Threshold the image to get binary smoke representation
    binary_image = grayscale_image.point(lambda p: p > threshold and 255)

    # Get the bounding box of the non-zero pixels (i.e., the smoke)
    bbox = binary_image.getbbox()

    if bbox:
        # Crop the original smoke image to the bounding box
        bounding_boxes_image = smoke_image.crop(bbox)
        return bounding_boxes_image
    else:
        return None

def rotate_image(image, max_rotation_angle):
    """
    Randomly rotates the image within the specified maximum rotation angle.

    Args:
        image (Image): The image as a PIL Image object.
        max_rotation_angle (float): The maximum rotation angle in degrees.

    Returns:
        Image: The rotated image as a PIL Image object.
    """
    rotation_angle = random.uniform(0, max_rotation_angle)
    rotated_image = image.rotate(rotation_angle, resample=Image.BICUBIC, expand=True)
    return rotated_image

def create_binary_mask(smoke_image):
    """
    Creates a binary mask from the smoke image.

    Args:
        smoke_image (Image): The smoke image as a PIL Image object (RGBA).

    Returns:
        Image: A binary mask image with the smoke in white and the rest in black.
    """
    # Extract the alpha channel and the RGB channels
    alpha_channel = smoke_image.getchannel('A')
    rgb_channels = smoke_image.convert('RGB')
    
    # Convert RGB channels to grayscale
    gray_channels = rgb_channels.convert('L')
    
    # Convert alpha channel and grayscale to numpy arrays
    alpha_array = np.array(alpha_channel)
    gray_array = np.array(gray_channels)
    
    # Apply Otsu's thresholding to alpha channel and grayscale
    alpha_threshold = threshold_otsu(alpha_array)
    gray_threshold = threshold_otsu(gray_array)
    
    # Create binary masks based on thresholding
    alpha_binary_mask = alpha_array >= alpha_threshold
    gray_binary_mask = gray_array >= gray_threshold
    
    # Combine the masks using logical AND operation
    combined_mask = np.logical_and(alpha_binary_mask, gray_binary_mask)
    
    # Convert the combined mask array back to PIL Image
    combined_mask_image = Image.fromarray(combined_mask.astype(np.uint8) * 255, 'L')

    return combined_mask_image

def calculate_random_position(background_width, background_height, smoke_width, smoke_height, background_image):
    """
    Calculates a random position to paste the smoke image within the background.

    Args:
        background_width (int): Width of the background image.
        background_height (int): Height of the background image.
        smoke_width (int): Width of the smoke image.
        smoke_height (int): Height of the smoke image.
        background_image (Image): The background image as a PIL Image object.

    Returns:
        tuple: A tuple containing the random x and y offsets for pasting the smoke image.
    """
    max_x_offset = background_width - smoke_width
    max_y_offset = background_height - smoke_height

    for _ in range(100):  # Try a maximum of 100 times
        x_offset = random.randint(0, max_x_offset)
        y_offset = random.randint(0, max_y_offset)

        if is_non_sky_region(background_image, x_offset, y_offset, smoke_width, smoke_height):
            return x_offset, y_offset
    
    # Return a fallback position if no valid position is found
    return max_x_offset, max_y_offset
    
def is_non_sky_region(image, x_offset, y_offset, width, height):
    """
    Checks if the given region is not part of the sky region based on certain color thresholds.

    Args:
        image (Image): The image containing the region to check.
        x_offset (int): X offset of the region.
        y_offset (int): Y offset of the region.
        width (int): Width of the region.
        height (int): Height of the region.

    Returns:
        bool: True if the region is not part of the sky region, False otherwise.
    """
    blue_threshold = 100
    whitish_threshold = 170

    pixel = image.getpixel((x_offset , y_offset ))
    if (
            pixel[2] > blue_threshold and pixel[2] - pixel[1] > 15 and pixel[2] - pixel[0] > 15
        ) or (
            pixel[0] > whitish_threshold and pixel[1] > whitish_threshold and pixel[2] > whitish_threshold
        ):
            return False
    return True
    """
    for i in range(width):
        for j in range(height):
            pixel = image.getpixel((x_offset , y_offset + j))
            if (
                pixel[2] > blue_threshold and pixel[2] - pixel[1] > 15 and pixel[2] - pixel[0] > 15
            ) or (
                pixel[0] > whitish_threshold and pixel[1] > whitish_threshold and pixel[2] > whitish_threshold
            ):
                return False
    return True
    """

def add_white_mask(image, alpha):
    """
    Adds a white overlay to the RGB channels of the input image.

    Args:
        image (PIL.Image.Image): The input image type RGB
        alpha (float): The alpha value for blending the white overlay. Should be in the range [0, 1].

    Returns:
        PIL.Image.Image: The image with the white overlay added.
    """
    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Create a white overlay with the same shape as the image
    overlay = np.ones_like(image_array) * [255, 255, 255]

    # Blend the original image with the white overlay using alpha
    blended_array = (1 - alpha) * image_array + alpha * overlay

    # Convert the blended array back to a PIL image
    blended_image = Image.fromarray(np.uint8(blended_array))

    return blended_image

def adjust_brightness(image, brightness_factor=1.0):
    """
    Adjusts the brightness of an image using PIL (Pillow).

    Args:
        image (PIL.Image): The input image as a PIL Image object.
        brightness_factor (float): Brightness adjustment factor (1.0 for no change).

    Returns:
        PIL.Image: The adjusted image as a PIL Image object.
    """
    enhancer = ImageEnhance.Brightness(image)
    adjusted_image = enhancer.enhance(brightness_factor)
    return adjusted_image

def gamma_correction(image, gamma=1.0):
    """
    Adjusts the brightness of the image using gamma correction.

    Args:
        image (PIL.Image): The image as a pil object
        gamma (float): Gamma correction factor (1.0 for no change).

    Returns:
        np.ndarray: The adjusted image as a NumPy array.
    """
    image = np.array(image)
    # Ensure the image is in the correct data type
    image = image.astype(np.float32) / 255.0

    # Apply gamma correction
    adjusted_image = np.power(image, gamma)

    # Rescale to 0-255 and convert back to uint8
    adjusted_image = (adjusted_image * 255.0).clip(0, 255).astype(np.uint8)

    return Image.fromarray(adjusted_image)

def composite_smoke(background_path, smoke_image_path,rescaling_factor=None,white_mask = (0.15,0.25),brightness_factor = None ,gamma_factor = 1 ):
    """
    Composites a smoke image onto a random background image with rotation and brightness adjustment.

    Args:
        background_path (str): Path to the background image.
        smoke_image_path (str): Path to the smoke image.
    Returns:
        Image: The composite image as a PIL Image object, or None if the smoke image is not found.
    """
    # Define rescaling and brightness parameters
    if rescaling_factor == None :
        rescaling_factor = random.uniform(0.12,0.33)
    if brightness_factor == None:
        brightness_factor = random.uniform(1.1,2)
    alpha = random.uniform(white_mask[0],white_mask[1])
    #print(rescaling_factor,brightness_factor)
    
    # Load the background image
    background = Image.open(background_path).convert("RGBA")
    background_width, background_height = background.size

    if os.path.exists(smoke_image_path):
        # Load the smoke image
        smoke_image = Image.open(smoke_image_path)
        
        smoke_image = adjust_brightness(image=smoke_image,brightness_factor=brightness_factor)
        smoke_image = gamma_correction(image=smoke_image,gamma=gamma_factor)
        # Get the bounding boxes of the smoke
        bounding_boxes_image = get_bounding_boxes(smoke_image)

        if bounding_boxes_image:
            # Calculate the maximum rescaling size based on the bounding box dimensions
            min_rescaling_size = min(background_width / bounding_boxes_image.width, background_height / bounding_boxes_image.height)
            if min_rescaling_size > 1 : # case where the bb is small
                smoke_rescaling_size = rescaling_factor
            else :  # case where the bb is large
                smoke_rescaling_size = min_rescaling_size*rescaling_factor

            # Resize the bounding boxed smoke image while maintaining the aspect ratio
            new_width = int(bounding_boxes_image.width * smoke_rescaling_size)
            new_height = int(bounding_boxes_image.height * smoke_rescaling_size)
            smoke_image = bounding_boxes_image.resize((new_width, new_height), Image.LANCZOS)

            # Create a transparent background for the smoke
            transparent_background = Image.new('RGBA', (background_width, background_height), (0, 0, 0, 0))

            # Calculate the position to paste the smoke image at the center of the background
            # Calculate the maximum allowed x and y offsets to ensure the smoke image is fully contained
            x_offset,y_offset = calculate_random_position(background_width,background_height,new_width,new_height,background_image=background)
            # Paste the brightness-adjusted smoke image onto the transparent background
            transparent_background.paste(smoke_image,(x_offset,y_offset),mask=smoke_image)
            # Create a Binary Mask
            binary_mask = create_binary_mask(transparent_background)
            # Composite the smoke onto the background
            composite = Image.alpha_composite(background, transparent_background)
            # Convert to RGB to ensure code stability
            composite = composite.convert("RGB")
            composite = add_white_mask(composite,alpha=alpha)
            return composite,binary_mask
        else:
            print("No bounding boxes found in the smoke image.")
            return None
    else:
        print("Smoke image not found.")
        return None

##-----------------------------------------------------------------------------------------
##                    methods for Selecting Background and Smoke Images
##-----------------------------------------------------------------------------------------

def select_background_image(base_folder):
    """
    Selects a random background image from the 'background_images' folder.
    Args:
        base_folder (str): The base folder of the project.
    Returns:
        str: The path to the selected background image, or None if no images are found.
    """
    # Construct the path to the background images folder
    # file number to select background data
    new_file_number = 0
    background_folder = os.path.join(base_folder, "background_images")
    # Get a list of all background image files with allowed extensions
    background_images = [f for f in os.listdir(background_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and int(f.split('_')[-1].split(".")[0])>=new_file_number]
    
    if background_images:
        # Randomly select a background image from the list
        background_path = os.path.join(background_folder, random.choice(background_images))
        return background_path
    else:
        print("No background images found.")
        return None

def select_smoke_image(base_folder):
    """
    Selects a random smoke image from the 'blender_images' subfolders.
    Args: -- base_folder (str): The base folder of the project.
    Returns -- (str) The path to the selected smoke image, or None if no images are found.
    """
    # Construct the path to the smoke images folder
    smoke_folder = os.path.join(base_folder, "blender_images")
    # Get a list of all subfolders with smoke plume images
    smoke_subfolders = [f for f in os.listdir(smoke_folder) if f.lower().startswith('smokeplume_')]
    # Uncomment to select specific smoke plume
    #smoke_subfolders = [f for f in smoke_subfolders if f.lower().startswith('smokeplume_88')]
    if smoke_subfolders:
        # Collect all smoke image paths from all subfolders
        smoke_image_paths = []
        for subfolder in smoke_subfolders:
            subfolder_path = os.path.join(smoke_folder, subfolder)
            smoke_images = [os.path.join(subfolder_path, image) for image in os.listdir(subfolder_path)]
            smoke_image_paths.extend(smoke_images)
        
        if smoke_image_paths:
            # Randomly select a smoke image path
            smoke_image_path = random.choice(smoke_image_paths)
            return smoke_image_path
        """
        # Randomly select a subfolder for smoke images
        random_subfolder = random.choice(smoke_subfolders)
        smoke_images_folder = os.path.join(smoke_folder, random_subfolder)
        # Get a list of all smoke image files with allowed extensions
        smoke_images = [f for f in os.listdir(smoke_images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if smoke_images:
            # Randomly select a smoke image from the list
            smoke_image_path = os.path.join(smoke_images_folder, random.choice(smoke_images))
            return smoke_image_path
        """
    print("No smoke images found.")
    return None

