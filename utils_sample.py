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