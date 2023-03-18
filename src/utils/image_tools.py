import os
import cv2
import torch
import numpy as np
import PIL.Image as Image
import torchvision.transforms.functional as F

def normalize(
    x: torch.Tensor,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
) -> torch.Tensor:
    """Normalizes the given image tensor channel-wise.

    Takes a tensor of the shape (C, H, W) or (B, C, H, W) and normalizes
    it based on the given channel-wise mean and standard deviation. If
    the mean or the standard deviation are not provided, the input is
    assumed to have 3 channels and is normalized using the standard
    values from albumentations library.

    See also:
        :func:`~utils.image_tools.unnormalize`.

    Args:
        x: The input tensor representing an image It can be both batched
            and unbatched. The values should be in the range [0, 1].
        mean: Sequence of means for each channel. Defaults to
            [0.485, 0.456, 0.406].
        std: Sequence of standard deviations for each channel. Defaults
            to [0.229, 0.224, 0.225].

    Returns:
        The normalized torch tensor.
    """
    return F.normalize(x, mean=mean, std=std)

def unnormalize(
    x: torch.Tensor,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
) -> torch.Tensor:
    """Unnormalizes the given image tensor channel-wise.
    
    Takes a tensor of the shape (C, H, W) or (B, C, H, W) and
    unnormalizes it based on the given channel-wise mean and standard
    deviation. If the mean or the standard deviation are not provided,
    the input is assumed to have 3 channels and is unnormalized using
    the standard values from albumentations library.

    See also:
        :func:`~utils.normalize`.

    Args:
        x: The input tensor representing an image It can be both batched
            and unbatched. The values should be any real numbers.
        mean: Sequence of means for each channel that was used to
            normalize them. Defaults to [0.485, 0.456, 0.406].
        std: Sequence of standard deviations for each channel that was
            used to normalize them. Defaults to [0.229, 0.224, 0.225].

    Returns:
        The unnormalized torch tensor.
    """
    
    # Create a torch tensor of mean and standard deviation to apply
    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    std = torch.tensor(std, device=x.device, dtype=x.dtype).view(-1, 1, 1)

    if x.ndim == 4:
        # If it is batched
        mean.unsqueeze_(0)
        std.unsqueeze_(0)

    return x * std + mean

def image_to_tensor(
    image: np.ndarray | Image.Image,
    is_batched: bool = False,
    device: str = "cpu",
) -> torch.Tensor:
    """Converts an image to a PyTorch tensor.

    Takes a PIL image or an image represented as a numpy array and
    converts it to a torch tensor.

    See also:
        :func:`~utils.image_tools.tensor_to_image`.

    Args:
        image: The image to convert. The expected shape is (H, W, 3)
            for RGB images or (H, W) for grayscale images. The values
            should be in range [0, 255].
        is_batched: Whether to add an extra dimension at the start of
            the tensor to represent the converted image as batch of size
            1. Defaults to False.
        device: The device to load the image on. Defaults to "cpu".

    Returns:
        A converted image. The output shape is (3, H, W) if the image
        was in RGB or (1, H, W) if it was in grayscale. If `is_batched`
        was set to True, then there is an extra dimension at the start
        resulting in shape (1, 3, H, W) or (1, 1, H, W).
    """
    # Use torchvision to_tensor method    
    image = F.to_tensor(image).to(device)

    if is_batched:
        # Add an extra dimension
        image = image.unsqueeze(0)
    
    return image

def tensor_to_image(
    tensor: torch.Tensor,
    convert_to_grayscale: bool = False,
    as_pil: bool = False,
) -> np.ndarray | Image.Image:
    """Converts a tensor to an image.

    Takes a torch tensor and converts it to a PIL image or an image
    represented as a numpy array.

    Note:
        The values in the tensor are all clipped to be in range [0, 1]
        before the conversion.
    
    See also:
        :func:`~utils.image_tools.image_to_tensor`.

    Args:
        tensor: The tensor to convert. The expected shape
            is (3, H, W) for RGB images and (1, H, W) or (H, W) for
            grayscale images. Also batches of size 1 are accepted, i.e.,
            (1, 3, H, W) and (1, 1, H, W).
        convert_to_grayscale: Whether to convert the
            image to grayscale if it has 3 channels. This has no effect
            if the image already has 1 channel. Defaults to False.
        as_pil: Whether to return the converted image
            as PIL Image.Image type. Defaults to False.

    Returns:
        The converted tensor to an image represented as a numpy array
        or PIL image. The output shape is (H, W, 3) or (H, W) for RGB or
        grayscale images, respectively and the value range is [0, 255].
    """
    # Clip the values and unnormalize with to_pil_image
    tensor = tensor.clip(0, 1).squeeze()
    image = F.to_pil_image(tensor)

    if convert_to_grayscale:
        # Convert to grayscale
        image = image.convert('L')

    if not as_pil:
        # Convert to numpy array
        image = np.array(image)
    
    return image

def load_image(
    path: str | os.PathLike,
    convert_to_grayscale: bool = False,
    as_tensor: bool = False,
    **tensor_kwargs,
) -> np.ndarray | torch.Tensor:
    """Loads an image from the specified path.

    Loads an image from the specified path, optionally converting it to
    a grayscale image and a torch tensor. If you want to load the image
    as PIL type, simply use :py:meth:`~PIL.Image.open`.

    Note:
        If the original image in the specified path has only one
        channel, i.e., is grayscale, then it will be loaded with 3
        replicated channels, unless `convert_to_grayscale` is specified
        as True.
    
    See also:
        :func:`~utils.image_tools.save_image`.

    Args:
        path: The path to the image.
        convert_to_grayscale: Whether to convert the loaded image to
            grayscale. Defaults to False.
        as_tensor: Whether to convert the loaded image to a torch
            tensor. Defaults to False.
        **tensor_kwargs: Extra arguments to convert the loaded image to
            a torch tensor. For more details, see the original function 
            :func:`~utils.image_tools.tensor_to_image`.

    Returns:
        The loaded image as a numpy array of shape (H, W, 3) if the
        image is RGB or (H, W) if it is grayscale, The values are in
        range [0, 255]. If `as_tensor` was specified to True, then a
        tensor will be returnd. For more details - see the original
        function :func:`~utils.image_tools.tensor_to_image`.
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

def save_image(
    image: torch.Tensor | np.ndarray | Image.Image,
    path: str | os.PathLike = "my_image.jpg",
    convert_to_grayscale: bool = False,
):
    """Saves the image as JPG to the specified path.

    Takes an image in any format and saves it to the specified path. If
    the path is not provided, then it saves the image to the current
    directory with the name "my_image.jpg".

    See also:
        :func:`~utils.image_tools.load_image`.

    Args:
        image: The image to save. If it is of type `torch.Tensor`, see
            :func:`utils.image_tools.tensor_to_image` for the expected
            format. If it is of type `np.ndarray`, the expected shape is
            (H, W, C) or (H, W) if the image is RGB or grayscale,
            respectively, and the value range should be [0, 255].
        path: The path to save the image. If the path does not end with
            `.jpg`, the extension is replaced automatically. Defaults
            to "my_image.jpg".
        convert_to_grayscale: Whether the image should be converted to
            grayscale. If True, then the saved image will only have 1
            channel. This has no effect if the provided image is already
            grayscale. Defaults to False.
    """
    if isinstance(image, torch.Tensor):
        # Convert properly from tensor
        image = tensor_to_image(image, as_pil=True)
    elif isinstance(image, np.ndarray):
        # In-built from array method
        image = Image.fromarray(image)
    
    if convert_to_grayscale:
        # Convert to grayscale
        image = image.convert('L')
    
    if not path.endswith(".jpg"):
        # Ensure correct format - change to JPG
        path = os.path.splitext(path)[0] + ".jpg"
    
    image.save(path)
