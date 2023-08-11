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
    max_x_offset = max(0, background_width - smoke_width)
    max_y_offset = max(0, background_height - smoke_height)

    for _ in range(100):  # Try a maximum of 100 times
        x_offset = random.randint(0, max_x_offset)
        y_offset = random.randint(0, max_y_offset)

        is_non_sky_x = all([is_non_sky_pixel(background_image.getpixel((x_offset + i, y_offset))) for i in range(smoke_width)])
        is_non_sky_y = all([is_non_sky_pixel(background_image.getpixel((x_offset, y_offset + i))) for i in range(smoke_height)])

        if  is_non_sky_x and is_non_sky_y:
            return x_offset, y_offset
    
    # Return a fallback position if no valid position is found
    return max_x_offset, max_y_offset

def is_non_sky_pixel(pixel):
    """
    Checks if the given pixel is not part of the sky region based on certain color thresholds.

    Args:
        pixel (tuple): The RGB values of the pixel.

    Returns:
        bool: True if the pixel is not part of the sky region, False otherwise.
    """
    # Define thresholds for blue and whitish colors
    blue_threshold = 100
    whitish_threshold = 170

    # Check if the pixel is not predominantly blue or whitish
    return not ((pixel[2] > blue_threshold and pixel[2] - pixel[1] > 15 and pixel[2] - pixel[0] > 15) or
                (pixel[0] > whitish_threshold and pixel[1] > whitish_threshold and pixel[2] > whitish_threshold))

def add_white_mask(image, alpha):
    # Create a white transparent overlay with the same size as the image
    overlay = Image.new('RGBA', image.size, color=(255, 255, 255, int(255 * alpha)))
    image=image.convert('RGBA')
    # Blend the original image with the white overlay using the blend() function
    blended_image = Image.blend(image, overlay, alpha)

    return blended_image

def composite_smoke(background_path, smoke_image_path,white_mask = (0.15,0.25),binary_mask_treshold = 10 ):
    """
    Composites a smoke image onto a random background image with rotation and brightness adjustment.

    Args:
        background_path (str): Path to the background image.
        smoke_image_path (str): Path to the smoke image.
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
            smoke_rescaling_size = random.uniform(0.2, 0.4) * max_rescaling_size

            # Resize the bounding boxed smoke image while maintaining the aspect ratio
            new_width = int(bounding_boxes_image.width * smoke_rescaling_size)
            new_height = int(bounding_boxes_image.height * smoke_rescaling_size)
            smoke_image = bounding_boxes_image.resize((new_width, new_height), Image.LANCZOS)

            # Create a transparent background for the smoke
            transparent_background = Image.new('RGBA', (background_width, background_height), (0, 0, 0, 0))

            # Calculate the position to paste the smoke image at the center of the background
            # Calculate the maximum allowed x and y offsets to ensure the smoke image is fully contained
            x_offset,y_offset = calculate_random_position(background_width,background_height,new_width,new_height,background)

            # Paste the brightness-adjusted smoke image onto the transparent background
            transparent_background.paste(smoke_image, (x_offset, y_offset), mask=smoke_image)
            # Create a Binary Mask
            binary_mask = create_binary_mask(transparent_background, threshold_value=binary_mask_treshold)
            # Composite the smoke onto the background
            composite = Image.alpha_composite(background, transparent_background)
            alpha = random.uniform(white_mask[0],white_mask[1])
            composite = add_white_mask(composite,alpha)
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
    background_folder = os.path.join(base_folder, "background_images")
    # Get a list of all background image files with allowed extensions
    background_images = [f for f in os.listdir(background_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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
    if smoke_subfolders:
        # Randomly select a subfolder for smoke images
        random_subfolder = random.choice(smoke_subfolders)
        smoke_images_folder = os.path.join(smoke_folder, random_subfolder)
        # Get a list of all smoke image files with allowed extensions
        smoke_images = [f for f in os.listdir(smoke_images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if smoke_images:
            # Randomly select a smoke image from the list
            smoke_image_path = os.path.join(smoke_images_folder, random.choice(smoke_images))
            return smoke_image_path
    print("No smoke images found.")
    return None

