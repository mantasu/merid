import os
import cv2
import json
import torch
import numpy as np

from PIL import Image
from typing import Any
from torchvision.transforms.functional import to_tensor, to_pil_image

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

def image_to_tensor(image: np.ndarray | Image.Image,
                    is_batched: bool = False,
                    device: str = "cpu"
                    ) -> torch.Tensor:
    """Converts an image to a tensor

    Takes a PIL image or an image represented as a numpy array and
    converts it to a torch tensor.

    Args:
        image (np.ndarray | Image.Image): The image to convert. The
            expected shape is (H, W, 3) for RGB images or (H, W) for
            grayscale images. The values should be in range [0, 255].
        is_batched (bool, optional): Whether to add an extra dimension
            at the start of the tensor to represent the converted image
            as batch of size 1. Defaults to False.
        device (str, optional): The device to load the image on.
            Defaults to "cpu".

    Returns:
        torch.Tensor: A converted image. The output shape is (3, H, W)
            if the image was in RGB or (1, H, W) if it was in grayscale.
            If `is_batched` was set to True, then there is an extra
            dimension at the start resulting in shape (1, 3, H, W) or
            (1, 1, H, W).
    """
    # Use torchvision to_tensor method    
    image = to_tensor(image).to(device)

    if is_batched:
        # Add an extra dimension
        image = image.unsqueeze(0)
    
    return image

def tensor_to_image(tensor: torch.Tensor,
                    convert_to_grayscale: bool = False,
                    as_pil: bool = False
                    ) -> np.ndarray | Image.Image:
    """Converts a tensor to an image

    Takes a torch tensor and converts it to a PIL image or an image
    represented as a numpy array.

    Note:
        The values in the tensor are all clipped to be in range [0, 1]
        before the conversion.

    Args:
        tensor (torch.Tensor): The tensor to convert. The expected shape
            is (3, H, W) for RGB images and (1, H, W) or (H, W) for
            grayscale images. Also batches of size 1 are accepted, i.e.,
            (1, 3, H, W) and (1, 1, H, W).
        convert_to_grayscale (bool, optional): Whether to convert the
            image to grayscale if it has 3 channels. This has no effect
            if the image already has 1 channel. Defaults to False.
        as_pil (bool, optional): Whether to return the converted image
            as PIL Image.Image type. Defaults to False.

    Returns:
        np.ndarray | Image.Image: The converted tensor to an image
            represented as a numpy array or PIL image. The output shape
            is (H, W, 3) or (H, W) for RGB or grayscale images,
            respectively and the value range is [0, 255].
    """
    # Clip the values and unnormalize with to_pil_image
    tensor = tensor.clip(0, 1).squeeze()
    image = to_pil_image(tensor)

    if convert_to_grayscale:
        # Convert to grayscale
        image = image.convert('L')

    if not as_pil:
        # Convert to numpy array
        image = np.array(image)
    
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
        image = tensor_to_image(image)
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



