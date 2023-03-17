import os
import sys
import tqdm
import numpy as np
import random

from skimage.morphology import binary_dilation, disk, binary_erosion

sys.path.append("src")

from PIL import Image
from architectures.remglass import RemGlass
from utils.config import parse_config
from utils.io import load_image, save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def process_celeba(model):
    os.makedirs("data/celeba/train/generated", exist_ok=True)
    os.makedirs("data/celeba/val/generated", exist_ok=True)
    os.makedirs("data/celeba/test/generated", exist_ok=True)

    transform = A.Compose([A.Normalize(), ToTensorV2()])

    for target in ["train", "val", "test"]:
        in_path = os.path.join("data/celeba", target, "glasses")
        out_path = os.path.join("data/celeba", target, "generated")

        for file in tqdm.tqdm(os.listdir(in_path)):
            img = transform(image=load_image(os.path.join(in_path, file)))["image"].to("cuda:0")[None, ...]

            out, mask = model(img)
            save_image(out, os.path.join(out_path, file[:-4] + "-gen-inpainted.jpg"))
            save_image(mask, path=os.path.join(out_path, file[:-4] + "-gen-mask.jpg"), is_grayscale=True)


def process_synthetic(model):
    os.makedirs("data/synthetic/train/generated", exist_ok=True)
    os.makedirs("data/synthetic/val/generated", exist_ok=True)
    os.makedirs("data/synthetic/test/generated", exist_ok=True)

    transform = A.Compose([A.Normalize(), ToTensorV2()])

    for target in ["train", "val", "test"]:
        in_path = os.path.join("data/synthetic", target, "glasses")
        out_path = os.path.join("data/synthetic", target, "generated")

        for file in tqdm.tqdm(os.listdir(in_path)):
            img = transform(image=load_image(os.path.join(in_path, file)))["image"].to("cuda:0")[None, ...]

            model.mask_retoucher.segment_when = "always"
            out, mask = model(img)
            save_image(out, os.path.join(out_path, file[:-4] + "-gen-inpainted-full.jpg"))
            save_image(mask, path=os.path.join(out_path, file[:-4] + "-gen-mask-full.jpg"), is_grayscale=True)

            model.mask_retoucher.segment_when = "never"
            out, mask = model(img)
            save_image(out, os.path.join(out_path, file[:-4] + "-gen-inpainted-frame.jpg"))
            save_image(mask, path=os.path.join(out_path, file[:-4] + "-gen-mask-frame.jpg"), is_grayscale=True)


def overlay(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    # Use the correct types
    mask = mask.astype(np.uint8)
    
    # Create the overlay colour
    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    # Init overlayed image
    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Create a completely overlayed image
        overlay = np.ones(image.shape) * np.array(colors[object_id])
        foreground = alpha * image + (1 - alpha) * overlay
        
        # Only keep the overlay on masked part
        binary_mask = mask == object_id
        im_overlay[binary_mask] = foreground[binary_mask]

        # Emphasize the contours in the segmented regions
        countours = binary_dilation(binary_mask, disk(4)) ^ binary_mask
        im_overlay[countours, :] = np.maximum(0, colors[0, :] - 5)
    
    # Compose an overlayed image
    overlayed = im_overlay.astype(image.dtype)

    return overlayed

def generate_syn_sunglass():
    os.makedirs("data/synthetic/train/sunglasses", exist_ok=True)
    os.makedirs("data/synthetic/val/sunglasses", exist_ok=True)
    os.makedirs("data/synthetic/test/sunglasses", exist_ok=True)

    random.seed(0)

    for target in ["train", "val", "test"]:
        in_path = os.path.join("data/synthetic", target, "glasses")
        out_path = os.path.join("data/synthetic", target, "sunglasses")

        for file in tqdm.tqdm(os.listdir(in_path)):
            img = load_image(os.path.join(in_path, file))
            mask = load_image(os.path.join("data/synthetic", target, "generated", file[:-4] + "-gen-mask-full.jpg"), convert_to_grayscale=True).squeeze() > 127

            if random.random() > 0.2:
                mask = binary_erosion(mask, disk(random.randint(2, 7)))

            if random.random() < 0.2:
                color = [random.randint(0, 30), random.randint(0, 30), random.randint(0, 30)]
                alpha=random.uniform(0.0, 0.2)
            else:
                color = [0, 0, 0]
                alpha=random.uniform(0.0, 0.1)
            

            image_sun = overlay(img, mask, colors=[color, color], cscale=1, alpha=alpha)
            save_image(image_sun, path=os.path.join(out_path, file.replace("-all", "-sunglasses")))      



if __name__ == "__main__":
    config = parse_config("configs/generate_preds.json")
    model = RemGlass(config).to("cuda:0")
    process_celeba(model)
    # process_synthetic(model)
    # generate_syn_sunglass()