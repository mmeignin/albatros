import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import harmonization_scripts.models as models 

##-----------------------------------------------------------------------------------------
##                        Methods for Image Harmonization
##-----------------------------------------------------------------------------------------

def harmonize_smoke_with_background(image, mask, model_path):
    """
    Harmonizes the smoke image with the background using a pre-trained model.

    Args:
        image (Image): The smoke image as a PIL Image object.
        mask (Image): The binary mask of the smoke image as a PIL Image object.
        model_path (str): Path to the pre-trained model checkpoint.

    Returns:
        Image: The harmonized image as a PIL Image object.
    """
    # Load the pre-trained model
    checkpoint_dict = torch.load(model_path, map_location=torch.device('cpu'))
    checkpoint = checkpoint_dict['state_dict']

    model = models.__dict__['rascv2']()
    model.load_state_dict(checkpoint)
    model.eval()

    # Preprocess the image and pass it through the model
    def preprocess_image(image, mask, input_size):
        original_size = image.size[::-1]  # PIL stores size as (width, height)
        image = image.convert('RGB')
        mask = mask.convert('L')
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])
        image = transform(image)
        mask = transform(mask)

        # Combine the image and mask along the channel dimension
        input_image = torch.cat((image, mask), dim=0).unsqueeze(0)

        return input_image, original_size

    # Set the desired input size for the model
    input_size = 1024

    # Preprocess the smoke image
    with torch.no_grad():
        input_image, original_size = preprocess_image(image, mask, input_size)
        output = model(input_image)

    # Convert the model's output to a numpy array
    output_np = (output[0] * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)

    # Convert the numpy array back to an RGB image using PIL
    output_image = Image.fromarray(output_np.transpose(1, 2, 0))

    return output_image
