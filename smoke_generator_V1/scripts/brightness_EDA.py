import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.image_composition import composite_smoke
from collections import defaultdict
import random

##-------------------------------------------------------------------------------------------------------------------
## Script to perform EDA on smoke images based on brightness extremums
##-------------------------------------------------------------------------------------------------------------------

# Function to calculate brightness
def calculate_brightness(image):
    grayscale_image = image.convert("L")
    pixel_values = list(grayscale_image.getdata())
    average_brightness = sum(pixel_values) / len(pixel_values)
    return average_brightness

# Function to analyze brightness of smoke images
def analyze_brightness(smoke_folder):
    brightness_data = defaultdict(list)

    for smoke_image_dir in os.listdir(smoke_folder):
        for smoke_image_file in os.listdir(os.path.join(smoke_folder, smoke_image_dir)):
            if smoke_image_file.endswith('.png'):
                smoke_image_path = os.path.join(smoke_folder, smoke_image_dir, smoke_image_file)
                smoke_image = Image.open(smoke_image_path)
                brightness = calculate_brightness(smoke_image)

                brightness_data['SmokeDirectory'].append(smoke_image_dir)
                brightness_data['ImageFile'].append(smoke_image_file)
                brightness_data['Brightness'].append(brightness)

    df_brightness = pd.DataFrame(brightness_data)
    return df_brightness

# Function to visualize composition based on brightness extremums
def visualize_composition(df, smoke_folder, background_path):
    # Calculate extremum values per Smoke Directory
    extremum_values = df.groupby('SmokeDirectory')['Brightness'].agg(['min', 'max'])

    for smoke_dir, (min_brightness, max_brightness) in extremum_values.iterrows():
        max_image_row = df[(df['SmokeDirectory'] == smoke_dir) & (df['Brightness'] == max_brightness)].iloc[0]

        max_image_path = os.path.join(smoke_folder, max_image_row['SmokeDirectory'], max_image_row['ImageFile'])
        composite_max, _ = composite_smoke(background_path, max_image_path, brightness_factor=random.uniform(1.2,1.4),gamma_factor=1)
        brightness_factor = random.uniform(1.2,1.4)
        composite_min, _ = composite_smoke(background_path, max_image_path, brightness_factor=1)
        composite_max, _ = composite_smoke(background_path, max_image_path, brightness_factor=brightness_factor,gamma_factor=1)

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(composite_min)
        plt.title(f"Min Brightness Image ({brightness_factor:.2f}{max_image_row['ImageFile']})")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(composite_max)
        plt.title(f"Max Brightness Image ({max_image_row['ImageFile']})")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# Example usage
smoke_folder = r'D:\mploi\Documents\Albatros\albatros\smoke_generator_V1\blender_images'
background_path = r'D:\mploi\Documents\Albatros\albatros\smoke_generator_V1\background_images\image_0387.png'

df_brightness = analyze_brightness(smoke_folder)
visualize_composition(df_brightness, smoke_folder, background_path)
