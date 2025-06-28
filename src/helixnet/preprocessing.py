import pathlib as pt
from typing import List, Dict
from PIL import Image
import numpy as np

def load_image(file_path:pt.Path):
    """
    This function loads the images in (channels, length, width) format
    in order to compatible with :class:`helixnet.layers.Conv2D`

    :param file_path: the path of the image to be loaded can a string or pathlib.Path object
    """
    with Image.open(file_path) as file:
        image = np.array(file).transpose(2, 0, 1)
    return image