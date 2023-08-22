import numpy as np
import cv2
import random 
from PIL import Image,ImageEnhance,ImageFilter,ImageDraw
import os 
import string


##-----------------------------------------------------------------------------------------
##                    Data Augmentation Transforms Methods
##-----------------------------------------------------------------------------------------

def apply_advanced_motion_blur(image, angle_degrees=np.random.randint(0,360), blur_strength=np.clip(np.random.exponential(4),0,20)):
    """
    Apply motion blur to an image using a specified kernel length and angle.

    Args:
        image (cv2 object): the input image.
        blur_strength (float): default is np.random.uniform(0,20)
        angle_degrees (float): Angle of the motion blur kernel in degrees. If None, a random angle is used.

    Returns:
        cv2 image : The image with motion blur applied.
    """
    # Convert angle to radians
    angle = np.deg2rad(angle_degrees) if angle_degrees is not None else np.deg2rad(np.random.uniform(0, 359))
    # Calculate kernel offsets using trigonometry
    kernel_size = int(blur_strength * 2) + 1
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        offset_x = int(center + i * np.cos(angle))
        offset_y = int(center + i * np.sin(angle))
        if 0 <= offset_x < kernel_size and 0 <= offset_y < kernel_size:
            kernel[offset_y, offset_x] = 1.0 / kernel_size

    # Apply the kernel using OpenCV's filter2D function
    motion_blur = cv2.filter2D(image, -1, kernel)

    # Calculate scaling factor to maintain brightness
    scaling_factor = np.sum(kernel)

    # Normalize image pixel values
    motion_blur = (motion_blur / scaling_factor).astype(np.uint8)

    return motion_blur

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

def apply_gaussian_blur(image, radius=2):
    """
    Apply Gaussian blur to an image using OpenCV.

    Args:
        image (array): the input image.
        radius (float): The radius of the Gaussian kernel. Default is 2.

    Returns:
        PIL.Image.Image: The image with Gaussian blur applied.
    """
    # Apply Gaussian blur using OpenCV
    blurred_image = cv2.GaussianBlur(image, (0, 0), radius)
    return blurred_image

def add_target_indicator(image):
    """Add a target indicator to an image.

    Args:
        image (cv2) : image

    Returns:
        PIL.Image.Image: Image with the target indicator added.
    """
    # Define the path to the target images
    target_relative_folder = r"target_images\\"
    target_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), target_relative_folder)
    # List all target image files
    target_files = os.listdir(target_folder)
    # Choose a random target image
    random_target = random.choice(target_files)
    target_path = os.path.join(target_folder, random_target)
    # Open the input image and create a result image with the same dimensions
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = pil_image.convert("RGBA")
    width, height = img.size
    result = Image.new('RGBA', (width, height))
    result.paste(img, (0, 0))
    # Define the target indicator color
    if random.choice([True, False]):
        color = (random.randint(50, 150), 255, random.randint(50, 150)) # greenish
    else:
        color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)) # whiteish
    # Open and resize the target image
    x_center = width // 2
    y_center = height // 2
    target = Image.open(target_path).convert("RGBA")
    if "target" in random_target:
        # resize target 
        target_size = random.randint(96,128)
        target = target.resize((target_size, target_size))
        # Calculate the position to paste the target in the center
        target_x = x_center - target_size // 2
        target_y = y_center - target_size // 2
        # Change the color of every pixel in the target image
        target_data = target.getdata()
        modified_target_data = [(color[0], color[1], color[2], pixel[3]) for pixel in target_data]
        target.putdata(modified_target_data)
        # Paste the modified target onto the result image
        result.paste(target, (target_x, target_y), target)
    cv2_result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    return cv2_result

def generate_random_text(length=10):
    """
    Generate a random text string with the specified length.

    Args:
        length (int): The length of the random text string.

    Returns:
        str: The randomly generated text string.
    """
    characters = string.ascii_letters + string.digits + string.punctuation + " "
    return ''.join(random.choice(characters) for _ in range(length))

def add_text_annotations(image):
    """
    Add meaningful text annotations to an image in the four corners using OpenCV.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The image with text annotations added.
    """
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    choice = random.choice([1,2,3])
    if choice == 1 :
        font_color = (255, 255, 255)  
    elif choice == 2 :
        font_color = (0, 0, 0)  
    else :
        font_color = (random.randint(0,100), 255, random.randint(0,100))  

    # Define meaningful text annotations with random content
    annotations = [
        f"Location: {generate_random_text(15)}",
        f"Date: {generate_random_text(15)}",
        f"Camera: {generate_random_text(15)}",
        f"Temperature: {generate_random_text(15)}"
    ]

    # Get image dimensions
    height, width, _ = img.shape

    # Define the margin from the corners
    margin = 10

    # Add random text annotations to the four corners
    cv2.putText(img, annotations[0], (margin, margin), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(img, annotations[1], (width - margin - 150, margin), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(img, annotations[2], (margin, height - margin), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(img, annotations[3], (width - margin - 150, height - margin), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return img



