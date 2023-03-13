import argparse
import torch

from utils.config import parse_config
from architectures.remglass import RemGlass

torch.set_float32_matmul_precision("medium")

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str,
                        default="configs/train_mask_inpainter.json",
                        help="The path to the JSON config file. Defaults " +\
                              "to 'configs/train_mask_inpainter.json'.")

    return parser.parse_args()

def get_example():
    import albumentations as A
    from utils.io import load_image
    from albumentations.pytorch import ToTensorV2

    images = [
        "data/celeba/val/glasses/162786.jpg",
        # "data/celeba/val/glasses/163521.jpg",
        "data/celeba/val/glasses/162938.jpg",
        # "data/celeba/val/glasses/182601.jpg",
        "data/synthetic/val/glasses/img-Glass001-401-1_neutral-3-industrial_pipe_and_valve_02-346-all.jpg",
        # "data/synthetic/val/glasses/img-Glass001-401-15_lip_funneler-0-balcony-115-all.jpg"
    ]

    transform = A.Compose([A.Normalize(), ToTensorV2()], additional_targets={f"image{i}": "image" for i in range(len(images))})

    images = [load_image(image) for image in images]
    images = transform(image=images[0], **{f"image{i}": img for i, img in enumerate(images)})
    images = torch.stack([images[f"image{i}"] for i in range(len(images)-1)], dim=0)

    return images



if __name__ == "__main__":
    config = parse_config(parse_arguments().config)
    remglass = RemGlass(config).to("cuda:0")
    x = get_example().to("cuda:0")

    from utils.augment import unnormalize
    
    mask_glasses, mask_shadows, out_glasses, out_shadows = remglass.forward_before_retoucher(x)
    mask, out_glasses, out_shadows, is_sunglasses = remglass.forward_before_inpainter(x)
    img_new = remglass(x)


    out_list = [unnormalize(x), mask_glasses, mask_shadows, mask, img_new]
    
    print(mask.shape)
    print(is_sunglasses.shape, is_sunglasses)
    print(out_glasses.shape)
    print(out_shadows.shape)
    print(mask_glasses.shape)
    print(mask_shadows.shape)
    

    import matplotlib.pyplot as plt
    from utils.convert import tensor_to_image

    for i in range(len(mask_glasses)):
        for j, out in enumerate(out_list):
            plt.subplot(len(mask_glasses), len(out_list), i * len(out_list) + j + 1)
            plt.imshow(tensor_to_image(out[i]))
    
    plt.show()

    

