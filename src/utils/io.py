import os
import cv2
import json
import torch
import numpy as np

from PIL import Image
from typing import Any

from .convert import image_to_tensor, tensor_to_image

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

def load_image(path: str | os.PathLike,
               convert_to_grayscale: bool = False,
               as_tensor: bool = False,
               **tensor_kwargs
               ) -> np.ndarray | torch.Tensor:
    """Loads an image from the specified path

    Loads an image from the specified path, optionally converting it to
    a grayscale image and a torch tensor. If you want to load the image
    as PIL type, simply use :py:meth:`~PIL.Image.open`.

    Note:
        If the original image in the specified path has only one
        channel, i.e., is grayscale, then it will be loaded with 3
        replicated channels, unless `convert_to_grayscale` is specified
        as True.

    Args:
        path (str | os.PathLike): The path to the image.
        convert_to_grayscale: (bool, optional): Whether to convert the
            loaded image to grayscale. Defaults to False.
        as_tensor (bool, optional): Whether to convert the loaded image
            to a torch tensor. Defaults to False.
        **tensor_kwargs: Extra arguments to convert the loaded image to
            a torch tensor. For more details, see the original function 
            :func:`here <io_and_types.tensor_to_image>`.

    Returns:
        np.ndarray | torch.Tensor: The loaded image as a numpy array of
            shape (H, W, 3) if the image is RGB or (H, W) if it is
            grayscale, The values are in range [0, 255]. If `as_tensor`
            was specified to True, then a tensor will be returnd - see
            :func:`here <io_and_types.tensor_to_image>` for more details
            about the expected output.
    """
    if convert_to_grayscale:
        # Read the image as grayscale (single channel)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        # Read RGB image (read image will always have 3 channels)
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    if as_tensor:
        # Convert to tensor (with the specified kwargs)
        image = image_to_tensor(image, **tensor_kwargs)
    
    return image

def save_image(image: torch.Tensor | np.ndarray | Image.Image,
               path: str | os.PathLike = "my_image.jpg",
               is_grayscale: bool = False):
    """Saves the image as JPG to the specified path

    Takes an image in any format and saves it to the specified path. If
    the path is not provided, then it saves the image to the current
    directory with the name "my_image.jpg".

    Args:
        image (torch.Tensor | np.ndarray | Image.Image): The image to
            save. If it is of type `torch.Tensor`, see the expected
            format :func:`here <io_and_types.tensor_to_image>`. If it
            is of type `np.ndarray`, the expected shape is (H, W, C) or
            (H, W) if the image is RGB or grayscale, respectively and
            the value range is expected to be [0, 255].
        path (str | os.PathLike, optional): The path to save the image.
            If the path does not end with `.jpg`, the extension is
            replaced automatically. Defaults to "my_image.jpg".
        is_grayscale (bool, optional): Whether the image should be
            converted to grayscale. If True, then the saved image will
            only have 1 channel. This has no effect if the provided
            image is already grayscale. Defaults to False.
    """
    if isinstance(image, torch.Tensor):
        # Convert properly from tensor
        image = tensor_to_image(image, as_pil=True)
    elif isinstance(image, np.ndarray):
        # In-built from array method
        image = Image.fromarray(image)
    
    if is_grayscale:
        # Convert to grayscale
        image = image.convert('L')
    
    if not path.endswith(".jpg"):
        # Ensure correct format - change to JPG
        path = os.path.splitext(path)[0] + ".jpg"
    
    image.save(path)



