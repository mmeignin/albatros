# image_harmonization.py
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import scripts.models as models
import matplotlib.pyplot as plt

def preprocess_image(image, mask, input_size):
    """
    Preprocess the input image and mask for the harmonization model.

    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the mask image.
        input_size (int): Desired input size for the model.

    Returns:
        torch.Tensor: Preprocessed input image as a tensor.
        tuple: Tuple containing the original size of the image (width, height).
    """
    image = image.convert('RGB')
    mask = mask.convert('L')

    original_size = image.size[::-1]  # PIL stores size as (width, height)

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    image = transform(image)
    mask = transform(mask)

    input_image = torch.cat((image, mask), dim=0).unsqueeze(0)

    return input_image, original_size

def harmonize_image(image, mask, model_dir,output_size):
    """
    Harmonize the input image using the spatial-separated attention module (s2am) model.

    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the mask image.
        model_dir (str): Path to the pre-trained s2am model checkpoint.
    Returns:
        PIL.Image.Image: Harmonized output image as a PIL Image object.
    """
    checkpoint_dict = torch.load(model_dir, map_location=torch.device('cpu'))
    checkpoint = checkpoint_dict['state_dict']

    model = models.__dict__['rascv2']()
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        input_image, original_size = preprocess_image(image, mask, output_size)
        output = model(input_image)

    output_np = (output[0] * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
    output_image = Image.fromarray(output_np.transpose(1, 2, 0))

    return output_image


