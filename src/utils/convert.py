import torch
import numpy as np

from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

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