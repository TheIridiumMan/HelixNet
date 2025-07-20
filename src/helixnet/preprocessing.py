import pathlib as pt
from typing import List, Dict

try:
    from PIL import Image
except ImportError:
    Image = None

import numpy as np


def load_image(file_path: pt.Path):
    """
    This function loads the images in (channels, length, width) format
    in order to compatible with :class:`helixnet.layers.Conv2D`

    :param file_path: the path of the image to be loaded can a string or pathlib.Path object
    """
    if not Image:
        raise ImportError("Can't import pillow run `pip install pillow`")

    with Image.open(file_path) as file:
        image = np.array(file).transpose(2, 0, 1)
    return image
