import pathlib as pt
from typing import List, Dict

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

def numpizing_img(img: Image.Image) -> np.ndarray:
    arr = np.zeros((img.width, img.height))
    img = img.convert("L")
    for i in range(img.width):
        for j in range(img.height):
            arr[i, j] = img.getpixel([j, i])
    return arr