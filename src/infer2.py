import torch
import argparse

import albumentations as A
from utils.io import load_image, tensor_to_image
from albumentations.pytorch import ToTensorV2

from utils.config import parse_config
from architectures.remglass import RemGlass
from architectures.mask_inpainter import MaskInpainter
from architectures.lafin.lafin_inpainter import LafinInpainter

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str,
                        default="configs/temp.json",
                        help="The path to the JSON config file. Defaults " +\
                              "to 'configs/train_mask_inpainter.json'.")

    return parser.parse_args()

def get_example():

    images = [
        # "data/celeba/val/glasses/162786.jpg",
        # "data/celeba/val/glasses/163521.jpg",
        # "data/celeba/val/glasses/162938.jpg",
        # "data/celeba/val/glasses/182601.jpg",
        "data/celeba/val/glasses/163610.jpg",
        # "data/celeba/test/glasses/182955.jpg",
        # "data/celeba/test/glasses/183090.jpg",
        # "data/synthetic/val/glasses/img-Glass001-407-8_jaw_right-1-cloud_layers-280-all.jpg",
        "data/synthetic/val/sunglasses/img-Glass001-407-9_jaw_forward-3-modern_buildings_night-155-sunglasses.jpg",
        # "data/synthetic/val/glasses/img-Glass001-413-1_neutral-2-carpentry_shop_02-198-all.jpg",
        "data/synthetic/val/glasses/img-Glass001-401-1_neutral-3-industrial_pipe_and_valve_02-346-all.jpg",
        "data/synthetic/val/glasses/img-Glass001-401-15_lip_funneler-0-balcony-115-all.jpg"
    ]

    transform = A.Compose([A.Normalize(), ToTensorV2()], additional_targets={f"image{i}": "image" for i in range(len(images))})

    images = [load_image(image) for image in images]
    images = transform(image=images[0], **{f"image{i}": img for i, img in enumerate(images)})
    images = torch.stack([images[f"image{i}"] for i in range(len(images)-1)], dim=0)

    return images


if __name__ == "__main__":
    config = parse_config(parse_arguments().config)
    remglass = RemGlass(config)
    x = get_example()

    from utils.augment import unnormalize
    
    with torch.no_grad():
        mask_glasses, mask_shadows, out_glasses, out_shadows = remglass.forward_before_retoucher(x)
        mask, out_glasses, out_shadows, is_sunglasses = remglass.forward_before_inpainter(x)
        inpainter = LafinInpainter(**{"det_weights": "checkpoints/landmark_detector.pth", "gen_weights": "checkpoints/InpaintingModel_gen.pth"})
        post_processer = MaskInpainter.load_from_checkpoint("checkpoints/unetplusplus-epoch=08-val_loss_mse=0.0021'.ckpt")
        img_inp = inpainter.forward(unnormalize(x), mask)

        imgs, tr = [], A.Compose([A.Normalize(), ToTensorV2()])
        for img in img_inp:
            imgs.append(tr(image=tensor_to_image(img))["image"])
        
        img_inp2 = torch.stack(imgs, dim=0)
        img_new1 = post_processer.post_processer(x, img_inp2, mask)
        img_new = post_processer.recolorizer(img_new1, x)



    out_list = [unnormalize(x), mask_glasses, mask_shadows, mask, img_inp, img_new1.clamp(0, 1).repeat(1, 3, 1, 1), img_new.clamp(0, 1)]
    
    # print(mask.shape)
    print(is_sunglasses.shape, is_sunglasses)
    # print(out_glasses.shape)
    # print(out_shadows.shape)
    # print(mask_glasses.shape)
    # print(mask_shadows.shape)
    

    import matplotlib.pyplot as plt
    from utils.convert import tensor_to_image

    for i in range(len(mask_glasses)):
        for j, out in enumerate(out_list):
            plt.subplot(len(mask_glasses), len(out_list), i * len(out_list) + j + 1)
            plt.imshow(tensor_to_image(out[i]))
    
    plt.show()
