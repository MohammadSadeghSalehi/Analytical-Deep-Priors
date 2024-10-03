import os
import numpy as np
from PIL import Image, ExifTags
import torch
import torchvision.transforms as transforms

def load_image_and_add_noise(index, x_size, y_size, channels, operator, noise_level):
    image_dir = './images'
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']  
    image_files = [f for f in sorted(os.listdir(image_dir)) if os.path.splitext(f)[1].lower() in valid_extensions]
    if index >= len(image_files):
        raise ValueError(f"Index {index} is out of range. Only {len(image_files)} images found.")
    
    image_path = os.path.join(image_dir, image_files[index])
    
    #Load the image and keep its orientation
    with Image.open(image_path) as img:
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = img._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation)
                if orientation_value == 3:
                    img = img.rotate(180, expand=True)
                elif orientation_value == 6:
                    img = img.rotate(270, expand=True)
                elif orientation_value == 8:
                    img = img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass  

        # Resize the image
        resize_transform = transforms.Resize((y_size, x_size))
        img = resize_transform(img)
        
        if channels == 1:
            img = img.convert('L')  
            to_tensor = transforms.ToTensor()
        elif channels == 3:
            img = img.convert('RGB')  
            to_tensor = transforms.ToTensor()
        else:
            raise ValueError("Unsupported number of channels. Only 1 (grayscale) or 3 (RGB) are supported.")
        
        img_tensor = to_tensor(img)  # Tensor of shape [C, H, W]

        # Apply the operator to the image
        if operator:
            blurred = operator(img_tensor)
        
        # Add Gaussian noise
        noise = torch.randn(blurred.size()) * noise_level
        noisy_img_tensor = blurred + noise
        
        # Clamp values to be within [0, 1]
        noisy_img_tensor = torch.clamp(noisy_img_tensor, 0, 1)
    
    # Return the original and noisy images
    return img_tensor.unsqueeze(0), noisy_img_tensor.unsqueeze(0)


    
    