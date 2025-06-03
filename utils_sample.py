import random
import numpy as np
from PIL import Image
import torch

def set_seed(seed: int):
    """
    Set the seed for reproducibility across different libraries and devices.
    
    Args:
        seed (int): The seed value to set.
    """
    # Set seed for Python's random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch CPU
    torch.manual_seed(seed)
    
    # Set seed for PyTorch GPU (if using CUDA)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Ensure deterministic results for CUDA operations (optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def resize_and_center_crop(image, target_size=512):
    w, h = image.size
    scale = target_size / min(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    image_cropped = image_resized.crop((left, top, right, bottom))

    return image_cropped


def resize_and_add_margin(image, target_size=512, background_color=(255, 255, 255)):
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    new_image = Image.new("RGB", (target_size, target_size), background_color)

    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    new_image.paste(image_resized, (left, top))

    return new_image
