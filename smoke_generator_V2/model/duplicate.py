import os
from PIL import Image
from imagehash import average_hash
import matplotlib.pyplot as plt

image_directory = r"D:\mploi\Documents\Albatros\albatros\smoke_generator_V2\real_images"

# Dictionary to store image properties
image_hashes = {}

# Traverse the directory and calculate image hashes
for root, dirs, files in os.walk(image_directory):
    for file in files:
        file_path = os.path.join(root, file)
        if os.path.isfile(file_path):
            with Image.open(file_path) as img:
                image_hash = str(average_hash(img))
                if image_hash in image_hashes:
                    image_hashes[image_hash].append(file_path)
                else:
                    image_hashes[image_hash] = [file_path]

# Filter out groups with more than one image (i.e., potential duplicates)
potential_duplicates = {k: v for k, v in image_hashes.items() if len(v) > 1}

# Display images for each group of potential duplicates
for group_num, (image_hash, file_paths) in enumerate(potential_duplicates.items(), start=1):
    print(f"Group {group_num}:")
    fig, axes = plt.subplots(1, len(file_paths), figsize=(12, 4))
    
    for ax, file_path in zip(axes, file_paths):
        with Image.open(file_path) as img:
            ax.imshow(img)
            ax.axis("off")
    
    plt.show()


