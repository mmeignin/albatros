import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
import os
##-----------------------------------------------------------------------------------------
##                        Post Processing of Smoke Images
##-----------------------------------------------------------------------------------------

def lower_smoke_density(smoke_image, alpha_factor=0.1):
    """
    Lower the density of a smoke image by reducing the alpha (opacity) values of smoke pixels based on their distance from the detected center of the smoke.

    Args:
        smoke_image (PIL.Image): The smoke image as a PIL Image.
        alpha_factor (float): The maximum reduction factor for alpha values (opacity). Should be in the range (0, 1).

    Returns:
        PIL.Image: The smoke image with lowered density (alpha values).
    """
    # Ensure that alpha_factor is in the valid range
    alpha_factor = max(0.0, min(1.0, alpha_factor))

    # Detect the center of the smoke and the largest contour
    center_x, center_y, max_distance = detect_smoke_center_and_max_distance(smoke_image)

    # Convert the smoke image to RGBA mode
    smoke_image = smoke_image.convert("RGBA")

    # Get the size of the smoke image
    width, height = smoke_image.size

    # Create a numpy array from the smoke image for efficient pixel access
    smoke_data = np.array(smoke_image)

    # Create a mask that represents the distance-based alpha reduction
    distance_mask = np.zeros((height, width), dtype=np.float32)

    for x in range(width):
        for y in range(height):
            # Calculate the distance from the pixel to the center
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            # Calculate the reduction factor based on distance
            reduction_factor = 1.0 - (distance / max_distance)**4 * (1.0 - alpha_factor)
            
            # Ensure that the reduction factor is within the range [0, 1]
            reduction_factor = max(0.0, min(1.0, reduction_factor))
            # Store the reduction factor in the distance mask
            distance_mask[y, x] = reduction_factor

    # Apply the distance-based alpha reduction to the smoke image
    smoke_data[:, :, 3] = (smoke_data[:, :, 3] * distance_mask).astype(np.uint8)

    # Convert the numpy array back to a PIL Image
    lowered_smoke_image = Image.fromarray(smoke_data)

    return lowered_smoke_image

def lower_smoke_density_exp(smoke_image, alpha_factor=1, exponent=0.1):
    """
    Lower the density of a smoke image by reducing the alpha (opacity) values of smoke pixels based on their distance from the detected center of the smoke.

    Args:
        smoke_image (PIL.Image): The smoke image as a PIL Image.
        alpha_factor (float): The maximum reduction factor for alpha values (opacity). Should be in the range (0, 1).
        exponent (float): The exponent for the exponential reduction function. Higher values make the reduction stronger.

    Returns:
        PIL.Image: The smoke image with lowered density (alpha values).
    """
    # Ensure that alpha_factor is in the valid range
    alpha_factor = max(0.0, min(1.0, alpha_factor))

    # Detect the center of the smoke and the largest contour
    center_x, center_y, max_distance = detect_smoke_center_and_max_distance(smoke_image)

    # Convert the smoke image to RGBA mode
    smoke_image = smoke_image.convert("RGBA")

    # Get the size of the smoke image
    width, height = smoke_image.size

    # Create a numpy array from the smoke image for efficient pixel access
    smoke_data = np.array(smoke_image)

    # Create a mask that represents the distance-based alpha reduction using an exponential function
    distance_mask = np.zeros((height, width), dtype=np.float32)

    for x in range(width):
        for y in range(height):
            # Calculate the distance from the pixel to the center
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            # Calculate the reduction factor based on distance using an exponential function
            reduction_factor = np.exp(-exponent * (distance / max_distance)) * (alpha_factor)

            # Ensure that the reduction factor is within the range [0, 1]
            reduction_factor = max(0.0, min(1.0, reduction_factor))

            # Store the reduction factor in the distance mask
            distance_mask[y, x] = reduction_factor

    # Apply the distance-based alpha reduction to the smoke image
    smoke_data[:, :, 3] = (smoke_data[:, :, 3] * distance_mask).astype(np.uint8)

    # Convert the numpy array back to a PIL Image
    lowered_smoke_image = Image.fromarray(smoke_data)

    return lowered_smoke_image

def detect_smoke_center_and_max_distance(smoke_image):
    """
    Detects the center of the smoke simulation within a smoke image using a mask and calculates the maximum distance
    from the center to the largest contour.

    Args:
        smoke_image (PIL.Image): The smoke image as a PIL Image.

    Returns:
        tuple: A tuple containing the x and y coordinates of the center, and the maximum distance.
    """
    # Convert the smoke image to grayscale
    smoke_gray = cv2.cvtColor(np.array(smoke_image), cv2.COLOR_RGBA2GRAY)

    # Threshold the grayscale image to create a binary mask
    _, mask = cv2.threshold(smoke_gray, 1, 255, cv2.THRESH_BINARY)

    # Find the contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assumed to be the smoke)
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the center of mass of the largest contour
        M = cv2.moments(largest_contour)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        # Calculate the maximum distance from the center to the contour
        max_distance = 0

        for point in largest_contour:
            distance = np.sqrt((point[0][0] - center_x)**2 + (point[0][1] - center_y)**2)
            max_distance = max(max_distance, distance)

        return center_x, center_y, max_distance

    # If no contours found, return the center of the image and a default max distance of 0
    return smoke_image.width // 2, smoke_image.height // 2, 0

def rotate_image(image, angle_degrees):
    """
    Rotate an image by a specified angle in degrees.

    Args:
        image (PIL.Image): The input image as a PIL Image.
        angle_degrees (float): The angle in degrees by which to rotate the image.

    Returns:
        PIL.Image: The rotated image.
    """
    # Ensure that angle_degrees is a valid float value
    angle_degrees = float(angle_degrees)

    # Open the input image
    img = image.copy()

    # Rotate the image by the specified angle
    rotated_image = img.rotate(angle_degrees, expand=True)

    return rotated_image

def display_center_and_max_distance(image, center_x, center_y, max_distance):
    """
    Display the center and the maximum distance from the center on the image.

    Args:
        image (PIL.Image): The input image as a PIL Image.
        center_x (int): The x-coordinate of the center.
        center_y (int): The y-coordinate of the center.
        max_distance (float): The maximum distance from the center.

    Returns:
        PIL.Image: The image with center and max distance displayed.
    """
    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Draw a red circle at the center
    circle_radius = max_distance
    draw.ellipse((center_x - circle_radius, center_y - circle_radius, center_x + circle_radius, center_y + circle_radius), outline="red")

    # Create a label for the maximum distance
    label = f"Max Distance: {max_distance:.2f}"

    # Load a font (you can adjust the font and size)
    font = ImageFont.load_default()

    label_x = center_x 
    label_y = center_y
    draw.text((label_x, label_y), label, fill="red", font=font)
    return image


if __name__ == '__main__':
    for j in range(0,100):
        input_image_relative_path = rf"..\blender_images\smokeplume_{j}"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        smoke_folder = os.path.join(script_dir, input_image_relative_path)
        max_idx = 101
        if os.path.exists(smoke_folder):
            for i in os.listdir(smoke_folder):
                smoke_image = Image.open(os.path.join(smoke_folder,i))
                # Example usage:
                # Detect the center of the smoke and the maximum distance from the center
                center_x, center_y, max_distance = detect_smoke_center_and_max_distance(smoke_image)

                # Display the center and max distance on the image
                image_with_info = display_center_and_max_distance(smoke_image.copy(), center_x, center_y, max_distance)

                # Show the image with the center and max distance
                #image_with_info.show()

                # Lower the smoke density with the first method (linear reduction) and display it
                lowered_smoke_image_linear = lower_smoke_density_exp(smoke_image,exponent=0.4)
                lowered_smoke_image_linear = rotate_image(lowered_smoke_image_linear,0)
                lowered_smoke_image_linear =  lowered_smoke_image_linear.transpose(Image.FLIP_LEFT_RIGHT)
                #lowered_smoke_image_linear.show()
                lowered_smoke_image_linear.save(os.path.join(smoke_folder,"00"+str(max_idx)+".png"))
                

                background_path = r"..\background_images\image_519.png"
                background = Image.open(os.path.join(script_dir,background_path)).convert('RGBA')
                #background.show()

                transparent_background = Image.new('RGBA', (background.width, background.height), (0, 0, 0, 0))
                transparent_background.paste(lowered_smoke_image_linear)
                #transparent_background.show()
                #transparent_background.save(os.path.join(smoke_folder,"00"+str(max_idx)+".png"))
                #composite = Image.alpha_composite(background,transparent_background )
                max_idx +=1
                #composite.show()
            