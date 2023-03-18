import torch
import argparse

import albumentations as A
from utils.io import load_image, tensor_to_image
from albumentations.pytorch import ToTensorV2

from utils.config import parse_config
from merid.recolorizer import Recolorizer
from colorize_data import ColorizeDataModule
from utils.augment import unnormalize
import matplotlib.pyplot as plt
from utils.image_tools import tensor_to_image
from infer2 import get_example, parse_arguments

from architectures.remglass import RemGlass
from architectures.lafin.lafin_inpainter import LafinInpainter


if __name__ == "__main__":
    x = get_example()
    config = parse_config(parse_arguments().config)
    remglass = RemGlass(config)
    mask, out_glasses, out_shadows, is_sunglasses = remglass.forward_before_inpainter(x)
    inpainter = LafinInpainter(**{"det_weights": "checkpoints/landmark_detector.pth", "gen_weights": "checkpoints/InpaintingModel_gen.pth"})
    img_inp = inpainter.forward(unnormalize(x), mask)



    cdm = ColorizeDataModule()
    batch = next(iter(cdm.test_dataloader()))
    batch[0] = torch.cat((batch[0], x), dim=0)
    batch[1] = torch.cat((batch[1], img_inp), dim=0)
    
    ref, target = batch
    ref, target = ref[::2], target[::2]
    gray = target.mean(dim=1, keepdim=True)

    
    
    with torch.no_grad():
        recolorizer = Recolorizer.load_from_checkpoint("checkpoints/resnet-epoch=04-val_loss=0.0001'.ckpt")
        coloured = recolorizer(gray, ref)
    

    out_list = [unnormalize(ref), gray.repeat(1, 3, 1, 1), coloured, target]

    for i in range(len(ref)):
        for j, out in enumerate(out_list):
            plt.subplot(len(ref), len(out_list), i * len(out_list) + j + 1)
            plt.imshow(tensor_to_image(out[i]))
    
    plt.show()