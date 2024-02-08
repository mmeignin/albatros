import numpy as np
import cv2
import random 
from PIL import Image,ImageEnhance
import os 
import string
import matplotlib.pyplot as plt

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

def add_interface(image):
    """Add a target indicator to an image.

    Args:
        image (np.ndarray) : Input image in NumPy array format (cv2.imread output).

    Returns:
        np.ndarray: Image with the target indicator added.
    """
    # Define the path to the target images
    target_relative_folder = r"interface_images\interface_images\\"
    target_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), target_relative_folder)

    # List all target image files
    target_files = os.listdir(target_folder)

    # Choose a random target image
    random_target = random.choice(target_files)
    target_path = os.path.join(target_folder, random_target)

    # Open and resize the target image
    interface = Image.open(target_path).convert("RGBA")
    interface = interface.resize(image.shape[1::-1])

    # Convert the input image to RGBA format
    image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    # Create a copy of the input image
    result = image.copy()

    # Paste the target indicator on the image
    x_center = result.shape[1] // 2 - interface.width // 2
    y_center = result.shape[0] // 2 - interface.height // 2

    # Convert the PIL Image to a NumPy array
    interface_np = np.array(interface)
    choice = random.choice([1,2,3])
    if choice == 1 :
        new_color = (255, 255, 255)  
    elif choice == 2 :
        new_color = (0, 0, 0)  
    else :
        new_color = (random.randint(0,100), 255, random.randint(0,100))  
    interface_np[:, :, :3] = new_color
    # Merge the interface image with the result image using alpha blending
    for i in range(interface_np.shape[0]):
        for j in range(interface_np.shape[1]):
            alpha = interface_np[i, j, 3] / 255.0
            result[y_center + i, x_center + j] = (
                (1 - alpha) * result[y_center + i, x_center + j] +
                alpha * interface_np[i, j, :3]
            )

    return result

def monochrone(image):
    # Define the color range for the sky
    lower_sky_color = np.array([100, 100, 100])
    upper_sky_color = np.array([255, 255, 255])

    # Define contrast and brightness adjustments
    alpha = 0.4  # Contrast control (1.0 - 3.0)
    beta = 0   # Brightness control (0 - 100)


    # Create a mask for the sky region
    sky_mask = cv2.inRange(image, lower_sky_color, upper_sky_color)

    # Darken the sky
    image[sky_mask > 0] = image[sky_mask > 0] - 10

    # Convert to grayscale
    gray_night_mode_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adjust contrast and brightness
    gray_night_mode_image = cv2.convertScaleAbs(gray_night_mode_image, alpha=alpha, beta=beta)

    return gray_night_mode_image

def add_flare(image):
    # Define the path to the target image (lens flare)
    flare_relative_path = r"flare_images\flare_1.png"
    flare_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), flare_relative_path)

    # Open and resize the lens flare image
    flare_image = Image.open(flare_path).convert("RGBA")
    flare_image = flare_image.resize(image.shape[1::-1])

    # Create a copy of the input image
    result = image.copy()

    # Paste the lens flare on the image
    x_center = result.shape[1] // 2 - flare_image.width // 2
    y_center = result.shape[0] // 2 - flare_image.height // 2

    # Convert the PIL Image to a NumPy array
    interface_np = np.array(flare_image)

    # Merge the interface image (lens flare) with the result image using alpha blending
    for i in range(interface_np.shape[0]):
        for j in range(interface_np.shape[1]):
            alpha = interface_np[i, j, 3] / 255.0
            result[y_center + i, x_center + j] = (    (1 - alpha) * result[y_center + i, x_center + j] + alpha * interface_np[i, j, :3])

    return result

def add_pixelisation(image):
    """
    Pixelize an image.

    Args:
        image (array): The input image (NumPy array).
    Returns:
        pixelized_image (array): Image with pixelization.
    """
    pixel_size = random.randint(7,15)
    # Get the image dimensions
    height, width = image.shape[:2]

    # Resize the image to a smaller size
    small_image = cv2.resize(image, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_NEAREST)

    # Resize it back to the original size to pixelize
    pixelized_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)

    return pixelized_image

def add_gaussian_noise(image, mean=0, std_dev=25):
    """
    Add Gaussian noise to an image.

    Args:
        image (array): The input image (NumPy array).
        mean (int): Mean value of the Gaussian noise.
        std_dev (int): Standard deviation of the Gaussian noise.

    Returns:
        noisy_image (array): Image with added Gaussian noise.
    """
    # Generate Gaussian noise with the same shape as the input image
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)

    # Add the noise to the image
    noisy_image = cv2.add(image, noise)

    # Ensure pixel values are within the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

def add_monochromatic_noise(image):
    """
    Add monochromatic noise to an image using one random color.

    Args:
        image (array): The input image (NumPy array).
        mean (int): Mean value of the noise.
        std_dev (int): Standard deviation of the noise.

    Returns:
        noisy_image (array): Image with added monochromatic noise.
    """
    mean = 0
    std_dev = 0.7
    # Generate a random color for the noise (R, G, B channels)
    noise_color = np.random.randint(0, 256, 3)

    # Generate monochromatic noise for each channel with the same color
    noise = np.random.normal(mean, std_dev, image.shape[:2]).astype(np.uint8)

    # Add the noise to each channel with the same color
    for channel in range(3):
        image[:, :, channel] += noise_color[channel] * noise

    # Ensure pixel values are within the valid range [0, 255]
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image

def adjust_brightness(image):
    """
    Adjust the brightness of an image while preserving color balance.

    Args:
        image (array): The input image (NumPy array).
        factor (float): Brightness adjustment factor (1.0 is the original, <1.0 darkens, >1.0 brightens).

    Returns:
        adjusted_image (array): Image with adjusted brightness.
    """
    # Convert the image to a floating-point representation
    image_float = image.astype(float)
    factor = random.uniform(0.5,1.5)
    # Separate the image into color channels
    b, g, r = cv2.split(image_float)

    # Apply the brightness adjustment factor to each color channel
    b = cv2.multiply(b, factor)
    g = cv2.multiply(g, factor)
    r = cv2.multiply(r, factor)

    # Merge the adjusted color channels back into the image
    adjusted_image = cv2.merge((b, g, r))

    # Ensure pixel values are within the valid range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    return adjusted_image

def adjust_color_balance(image):
    """
    Adjust the color balance of an image.

    Args:
        image (array): The input image (NumPy array).
        blue_balance (float): Blue channel balance (1.0 is the original).
        green_balance (float): Green channel balance (1.0 is the original).
        red_balance (float): Red channel balance (1.0 is the original).

    Returns:
        balanced_image (array): Image with adjusted color balance.
    """
    # Separate the image into color channels
    b, g, r = cv2.split(image)
    blue_balance = random.uniform(0.5,1.5)
    green_balance = random.uniform(0.5,1.5)
    red_balance = random.uniform(0.5,1.5)
    # Apply the color balance adjustments to each channel
    b = cv2.multiply(b, blue_balance)
    g = cv2.multiply(g, green_balance)
    r = cv2.multiply(r, red_balance)

    # Merge the adjusted color channels back into the image
    balanced_image = cv2.merge((b, g, r))

    # Ensure pixel values are within the valid range [0, 255]
    balanced_image = np.clip(balanced_image, 0, 255).astype(np.uint8)

    return balanced_image

def apply_cutout(image):
    """
    Apply Cutout augmentation to the input image.

    Parameters:
    - image: Input image to be augmented.

    Returns:
    - cutout_image: Augmented image with Cutout applied.
    """
    cutout_image = image.copy()
    
    # Set Cutout parameters
    cutout_size = np.random.randint(200, 350)  # Adjust the patch size as needed
    num_cuts = np.random.randint(1, 4)         # Adjust the number of cuts as needed
    
    gray = np.random.randint(0, 256)           # Generate a random gray value
    new_color = (gray, gray, gray)
    
    for _ in range(num_cuts):
        x, y = np.random.randint(0, image.shape[0] - cutout_size), np.random.randint(0, image.shape[1] - cutout_size)
        cutout_image[x:x+cutout_size, y:y+cutout_size, :] = new_color
    
    return cutout_image


