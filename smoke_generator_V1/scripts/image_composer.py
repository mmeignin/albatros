from PIL import Image, ImageEnhance, ImageOps
import os
import random
import csv

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

def adjust_brightness(image, brightness_range=(0.8, 1.2)):
    """
    Randomly adjusts the brightness of the image within the specified brightness range.

    Args:
        image (Image): The image as a PIL Image object.
        brightness_range (tuple, optional): The range of brightness adjustments as a tuple (min_brightness, max_brightness). Default is (0.8, 1.2).

    Returns:
        Image: The brightness-adjusted image as a PIL Image object.
    """
    brightness_factor = random.uniform(*brightness_range)
    brightness_adjusted_image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    return brightness_adjusted_image

def create_binary_mask(smoke_image, threshold_value=128):
    """
    Creates a binary mask from the smoke image.

    Args:
        smoke_image (Image): The smoke image as a PIL Image object.
        threshold_value (int, optional): The threshold value for creating the binary mask. Default is 128.

    Returns:
        Image: A binary mask image with the smoke in white and the rest in black.
    """
    # Create a mask image with the smoke in white and the rest in black
    mask_image = smoke_image.convert('L')

    # Apply thresholding to get a binary mask
    binary_mask = mask_image.point(lambda x: 255 if x > threshold_value else 0, '1')

    return binary_mask

def calculate_random_position(background_width, background_height, smoke_width, smoke_height):
    """
    Calculates a random position to paste the smoke image within the background.

    Args:
        background_width (int): Width of the background image.
        background_height (int): Height of the background image.
        smoke_width (int): Width of the smoke image.
        smoke_height (int): Height of the smoke image.

    Returns:
        tuple: A tuple containing the random x and y offsets for pasting the smoke image.
    """
    # Calculate the maximum allowed x and y offsets to ensure the smoke image is fully contained
    max_x_offset = max(0, background_width - smoke_width)
    max_y_offset = max(0, background_height - smoke_height)

    # Calculate the random position to paste the smoke image within the background
    x_offset = random.randint(0, max_x_offset)
    y_offset = random.randint(0, max_y_offset)
    return x_offset, y_offset

def randomly_select_subfolder(subfolders, weights):
    """
    Randomly selects a subfolder from a list of subfolders based on the given weights.

    Args:
        subfolders (list): List of subfolder names.
        weights (list): List of weights for each subfolder.

    Returns:
        str: The selected subfolder name.
    """
    return random.choices(subfolders, weights=weights)[0]

def composite_smoke_with_rotation(background_path, smoke_image_path, max_rotation_angle, brightness_range):
    """
    Composites a smoke image onto a random background image with rotation and brightness adjustment.

    Args:
        background_path (str): Path to the background image.
        smoke_image_path (str): Path to the smoke image.
        max_rotation_angle (float): Maximum rotation angle in degrees for the smoke image.
        brightness_range (tuple): Range of brightness adjustments as a tuple (min_brightness, max_brightness).

    Returns:
        Image: The composite image as a PIL Image object, or None if the smoke image is not found.
    """
    # Load the background image
    background = Image.open(background_path).convert("RGBA")
    background_width, background_height = background.size

    if os.path.exists(smoke_image_path):
        # Load the smoke image
        smoke_image = Image.open(smoke_image_path)

        # Get the bounding boxes of the smoke
        bounding_boxes_image = get_bounding_boxes(smoke_image)

        if bounding_boxes_image:
            # Calculate the maximum rescaling size based on the bounding box dimensions
            max_rescaling_size = min(background_width / bounding_boxes_image.width, background_height / bounding_boxes_image.height)

            # Randomly select a rescaling size within a certain range
            smoke_rescaling_size = random.uniform(0.4, 1) * max_rescaling_size

            # Resize the bounding boxed smoke image while maintaining the aspect ratio
            new_width = int(bounding_boxes_image.width * smoke_rescaling_size)
            new_height = int(bounding_boxes_image.height * smoke_rescaling_size)
            smoke_image = bounding_boxes_image.resize((new_width, new_height), Image.LANCZOS)

            # Randomly rotate the smoke image
            rotated_smoke = rotate_image(smoke_image, max_rotation_angle)

            # Randomly adjust the brightness of the smoke image
            brightness_adjusted_smoke = adjust_brightness(rotated_smoke, brightness_range)

            # Create a transparent background for the smoke
            transparent_background = Image.new('RGBA', (background_width, background_height), (0, 0, 0, 0))

            # Calculate the position to paste the smoke image at the center of the background
            x_offset = (background_width - brightness_adjusted_smoke.width) // 2
            y_offset = (background_height - brightness_adjusted_smoke.height) // 2

            # Paste the brightness-adjusted smoke image onto the transparent background
            transparent_background.paste(brightness_adjusted_smoke, (x_offset, y_offset), mask=brightness_adjusted_smoke)

            # Composite the smoke onto the background
            composite = Image.alpha_composite(background, transparent_background)

            # Return the composite image
            return composite
        else:
            print("No bounding boxes found in the smoke image.")
            return None
    else:
        print("Smoke image not found.")
        return None


# You can add more functions here as needed
