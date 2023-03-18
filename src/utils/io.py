import os
import cv2
import json
import torch
import numpy as np

from PIL import Image
from typing import Any

from .image_tools import image_to_tensor, tensor_to_image

def load_json(path: str | os.PathLike) -> dict[str, Any]:
    """Loads JSON file

    Simply reads the specified file that ends with ".json" and parses
    its contents to python values to make up a python dictionary.
    
    Args:
        path (str | os.PathLike): The path to .json file
    
    Returns:
        dict[str, Any]: A python dictionary
    """
    with open(path, 'r') as f:
        # Load JSON file to py
        py_dict = json.load(f)
    
    return py_dict



