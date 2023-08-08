import os
import random
from PIL import Image
from image_composer import composite_smoke_with_rotation

def select_background_image(base_folder):
    background_folder = os.path.join(base_folder, "background_images")
    background_images = [f for f in os.listdir(background_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if background_images:
        background_path = os.path.join(background_folder, random.choice(background_images))
        return background_path
    else:
        print("No background images found.")
        return None

def select_smoke_image(base_folder):
    smoke_folder = os.path.join(base_folder, "blender_images")
    smoke_subfolders = [f for f in os.listdir(smoke_folder) if f.lower().startswith('smokeplume_')]
    if smoke_subfolders:
        random_subfolder = random.choice(smoke_subfolders)
        smoke_images_folder = os.path.join(smoke_folder, random_subfolder)
        smoke_images = [f for f in os.listdir(smoke_images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if smoke_images:
            smoke_image_path = os.path.join(smoke_images_folder, random.choice(smoke_images))
            return smoke_image_path
    print("No smoke images found.")
    return None

def main():
    base_folder = r'D:\mploi\Documents\Albatros\albatros\smoke_generator_V1'

    # Select background image
    background_path = select_background_image(base_folder)
    if not background_path:
        return

    # Select smoke image
    smoke_image_path = select_smoke_image(base_folder)
    if not smoke_image_path:
        return

    # Composite smoke with rotation
    composite_image = composite_smoke_with_rotation(background_path, smoke_image_path, max_rotation_angle=30, brightness_range=(0.5, 1.5))
    if composite_image:
        composite_image.show()
    
    

if __name__ == "__main__":
    main()
