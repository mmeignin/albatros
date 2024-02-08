from Lens_flare.data_loader import Flare_Image_Loader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math,os,random

def add_lens_flare(image_path, flare_path):
    """
    Adds lens flares to images.

    Args:
        image_path (str): Path to the images.
        flare_path (str): Path to the lens flare assets.

    Returns:
        list: List of images with lens flares.
    """
    transform_base = transforms.Compose([transforms.Resize((1080, 1080))])

    transform_flare = transforms.Compose([transforms.RandomAffine(degrees=(0, 360), scale=(0.8, 1.5),
                            translate=(300/1440, 300/1440), shear=(-20, 20)),
                            transforms.CenterCrop((1080, 1080)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip()
                            ])

    flare_image_loader = Flare_Image_Loader(image_path, transform_base, transform_flare)
    images_with_flares = []
    for i in range(0,600,66):
        flare_type = random.choice(os.listdir(flare_path + '/Scattering_Flare'))
        print(flare_type)
        flare_image_loader.load_scattering_flare(flare_path, flare_path + '/Scattering_Flare/' + flare_type)
        flare_image_loader.load_reflective_flare(flare_path, flare_path + '/Reflective_Flare')
        _, _, test_merge_img, _ = flare_image_loader[i]
        images_with_flares.append(test_merge_img)

    return images_with_flares

# Replace this directory with your images directory
image_path = r'smoke_dataset_V1\images'
flare_path = r'smoke_generator_V1\scripts\utils\Lens_flare\Flare7Kpp\Flare7K'

# Uncomment this line to add lens flares to the images
images_with_flares = add_lens_flare(image_path, flare_path)

# Display the images with lens flares
# Create a subplot grid for displaying the images with lens flares
num_images = len(images_with_flares)
num_columns = 5 # You can change this to adjust the number of columns
num_rows = math.ceil(num_images / num_columns)

fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 3))

# Adjust the horizontal and vertical spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0.02)  # You can adjust wspace and hspace as needed

# Display the images with lens flares
for i, image in enumerate(images_with_flares, 1):
    # Calculate subplot position (row and column)
    row_idx = (i - 1) // num_columns
    col_idx = (i - 1) % num_columns

    # Display the image in the subplot
    ax = axs[row_idx, col_idx]
    ax.imshow(image.permute(1, 2, 0))
    ax.axis('off')

# Remove empty subplots if there are fewer images than expected
for i in range(num_images, num_rows * num_columns):
    axs.flatten()[i].remove()

# Save the composite image to a file
#plt.savefig("out.png")
plt.show()