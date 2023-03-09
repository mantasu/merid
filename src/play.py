import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from train import load_json, parse_arguments, prepare_model, prepare_datamodule
from nafnet2.nafnet import NAFNet

import cv2
import math
from torchvision.utils import make_grid

def tensor_to_image(tensor):
    if (tensor < 0).any():
        tensor = (tensor + 1.0) / 2.0
    
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    
    if tensor.shape[0] == 3:
        tensor = np.moveaxis(tensor, 0, -1)
    
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze()

    return Image.fromarray(tensor)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            elif img_np.shape[2] == 3:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

if __name__ == "__main__":
    config = load_json(parse_arguments().config)
    mask_generator = prepare_model("mask_generator", config["mask_generator"])
    datamodule = prepare_datamodule(config["data"])
    dataloader = datamodule.val_dataloader()

    it = iter(dataloader)
    x1, x2, y1, y2 = next(it)
    num_row = 2

    r = torch.rand_like(x2)
    x1 = (x1 + r).clip(-1, 1)
    print(x1.min(), x1.max())

    post_processer = NAFNet(width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    loaded = torch.load("checkpoints/NAFNet-SIDD-width32.pth")
    post_processer.load_state_dict(loaded["params"])

    o2 = post_processer((x1 + 1) / 2).detach()

    plt.figure(figsize=(20, 10))

    for i, img in enumerate(x1):
        img = tensor_to_image(img)
        out = tensor2img(o2[i], False)

        plt.subplot(num_row, len(x2), 0 * len(x2) + i + 1)
        plt.imshow(img)
        plt.subplot(num_row, len(x2), 1 * len(x2) + i + 1)
        plt.imshow(out)
    
    plt.show()