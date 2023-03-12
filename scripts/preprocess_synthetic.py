import os
import tqdm
import torch
import random
import argparse
import numpy as np
import torchvision.transforms as T

from PIL import Image
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

def parse_args() -> argparse.Namespace:
    # Instantiate arguments parser
    parser = argparse.ArgumentParser()
    
    # Define all the command line arguments and default values
    parser.add_argument("--img-dir", type=str,
        default="data/synthetic/ALIGN_RESULT_v2",
        help=f"Path to Synthetic images. Defaults to "
             f"'data/synthetic/ALIGN_RESULT_v2'.")
    parser.add_argument("--save-dir", type=str, default="data/synthetic",
        help=f"Path to save the processed data (data splits). Defaults to "
             f"'data/synthetic'.")
    parser.add_argument("--seed", type=int, default=0,
        help=f"The seed for randomly shuffling the data before splitting to "
             f"train/val/test. Defaults to 0.")
    parser.add_argument("--split-frac", nargs=2, type=float, default=[.1, .1],
        help=f"Two float numbers determining the validation and the test data "
             f"fractions of the whole dataset. Defaults to [0.1, 0.1].")
    
    return parser.parse_args()

def train_val_test_split(files: list[str], val_size: float = .1,
                         test_size: float = .1, seed: int = 0) \
                         -> list[tuple[str, str]]:
    # Create indices at which to split the full data
    tags = list({file.split('-')[2] for file in files})
    test_idx = len(tags) - round(len(tags) * test_size)
    val_idx = test_idx - round(len(tags) * val_size)

    # Set seed and shuffle
    random.seed(seed)
    random.shuffle(tags)

    # Init split map
    split_info = {}

    for i, tag in enumerate(tags):
        if i < val_idx:
            # If index less than val idx
            split_info[tag] = "train"
        elif i >= test_idx:
            # If index more than test index
            split_info[tag] = "test"
        else:
            # If index between val and test
            split_info[tag] = "val"
    
    return [(file, split_info[file.split('-')[2]]) for file in files]

def generate_shadow_mask(img_with_shadow: torch.Tensor, img_without_shadow: torch.Tensor):
    # Compute the shadow difference and generate its mask
    diff = torch.abs(img_with_shadow - img_without_shadow)
    diff = diff[0:1, :, :] * 0.3 + diff[1:2, :, :] * 0.59 + diff[2:3, :, :] * 0.11
    all_true = torch.ones(*diff.shape).to(diff.device)
    all_false = torch.zeros(*diff.shape).to(diff.device)
    label = torch.where(diff > 0.1, all_true, all_false)

    return label

def save_shadow_mask_label(tensor: torch.Tensor, save_path: str):
    # Convert to proper image type and save to path
    out = tensor.cpu().numpy().transpose(1, 2, 0)
    out = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)
    out_PIL = Image.fromarray(out[:, :, 0])
    out_PIL.save(save_path)

def worker(data_dir: str, save_dir: str, transform: T.Compose, file: tuple[str, str]):
    # Write a helper function for making paths, generate a list of images
    make_path = lambda x: os.path.join(data_dir, file[0].replace("-all", x))
    paths = [make_path(x) for x in ["-all", "-glass", "-face", "-seg"]]
    images = [Image.open(path) for path in paths]        

    # Write some more helper functions for creating filenames/paths
    make_name = lambda x: file[0].replace("-all", x).replace(".png", ".jpg")
    make_path = lambda x, y: os.path.join(save_dir, file[1], x, make_name(y))

    # Generate a shadow mask of 'all' and 'glass' images, then save it
    mask_shadow = generate_shadow_mask(*[transform(x) for x in images[:2]])
    save_shadow_mask_label(mask_shadow, make_path("masks", "-shseg"))
    
    # Save the remaining pictures in JPG format
    images[0].save(make_path("glasses", "-all"))
    images[2].save(make_path("no_glasses", "-face"))
    images[3].save(make_path("masks", "-seg"))

def walk_through_files(data_dir: str, save_dir: str, split_size: list[float, float], seed: int):
    # Filter a list of filenames such that they only include "-all"
    files = [*filter(lambda x: "-all.png" in x, os.listdir(data_dir))]
    transform = T.Compose([T.ToTensor(), T.Normalize([.5]*3, [.5]*3)])
    files = train_val_test_split(files, *split_size, seed)

    for split in ["train", "val", "test"]:
        for dirname in ["glasses", "no_glasses", "masks"]:
            # Create empty directories to save the image files from dataset
            os.makedirs(os.path.join(save_dir, split, dirname), exist_ok=True)
    
    # Create a worker wrapper function to execute in threads
    worker_fn = partial(worker, data_dir, save_dir, transform)

    with ThreadPool(cpu_count()) as pool:
        # Generate a pool map, pass it to progress bar and execute
        pool_map = pool.imap_unordered(worker_fn, files)
        list(tqdm.tqdm(pool_map, total=len(files)))

def main():
    # Get args & process
    args = parse_args()
    walk_through_files(args.img_dir, args.save_dir, args.split_frac, args.seed)

if __name__ == "__main__":
    main()
