import os
import tqdm
import argparse
import numpy as np
from PIL import Image

def parse_args() -> argparse.Namespace:
    # Instantiate arguments parser
    parser = argparse.ArgumentParser()
    
    # Define all the command line arguments and default values
    parser.add_argument("--img-dir",
        default="data/celeba-mask-hq/CelebAMask-HQ/CelebA-HQ-img",
        help=f"Path to CelebA Mask HQ images. Defaults to "
             f"'data/celeba-mask-hq/CelebAMask-HQ/CelebA-HQ-img'.")
    parser.add_argument("--mask-dir",
        default="data/celeba-mask-hq/CelebAMask-HQ/CelebAMask-HQ-mask-anno",
        help=f"Path to CelebA Mask HQ masks. Defaults to "
             f"'data/celeba-mask-hq/CelebAMask-HQ/CelebAMask-HQ-mask-anno'.")
    parser.add_argument("--save-dir", default="data/celeba-mask-hq",
        help=f"Path to save the processed data to (data splits). Defaults to "
             f"'data/celeba-mask-hq'.")
    parser.add_argument("--split-info-file-paths", nargs=3, default=\
        [f"data/celeba/{x}_label.txt" for x in ["train", "val", "test"]],
        help=f"3 paths to files that contain information on how to split the "
             f"data into 3 parts. Defaults to ['data/celeba/train_label.txt', "
             f"'data/celeba/val_label.txt', 'data/celeba/test_label.txt'].")
    parser.add_argument("--celeba-mapping-file-path",
        default="data/celeba-mask-hq/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt",
        help=f"The path to .txt file which maps CelebA Mask HQ pictures to "
             f"corresponding original CelebA filenames. Defaults to "
             f"'data/celeba-mask-hq/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt'.")
    parser.add_argument("--resize", nargs=2, type=int, default=[256, 256],
        help=f"The width and height to resize the images and the masks. "
             f"Defaults to [256, 256].")
    
    return parser.parse_args()

def generate_split_paths(split_info_file_paths: list[str, str, str],
                         celeba_mapping_file_path: str,
                         save_dir: str,
                         ) -> dict[int, str]:
    for split_info_path in split_info_file_paths:
        # Read the first column of the the data split info file (filenames)
        file_names = np.genfromtxt(split_info_path, dtype=str, usecols=0)

        # Determine the type of split
        if "train" in split_info_path:
            train_set = {*file_names}
            subdir = "train"
        elif "val" in split_info_path:
            val_set = {*file_names}
            subdir = "val"
        elif "test" in split_info_path:
            test_set = {*file_names}
            subdir = "test"

        # Create image and mask directories as well while looping
        os.makedirs(os.path.join(save_dir, subdir, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, subdir, "masks"), exist_ok=True)
    
    # Init split info
    split_info = {}
    
    with open(celeba_mapping_file_path, 'r') as f:
        for line in f.readlines()[1:]:
            # Read Celeba Mask HQ index and CelebA file name
            [idx, _, orig_file] = line.split()

            if orig_file in train_set:
                # If CelebA file name belongs to train dataset
                split_info[int(idx)] = os.path.join(save_dir, "train")
            elif orig_file in val_set:
                # If CelebA file name belongs to val dataset
                split_info[int(idx)] = os.path.join(save_dir, "val")
            elif orig_file in test_set:
                # If CelebA file name belongs to test dataset
                split_info[int(idx)] = os.path.join(save_dir, "test")

    return split_info

def walk_through_masks(mask_dir: str,
                       img_dir: str,
                       split_info: dict[int, str],
                       resize: list[int, int] = [256, 256]):
    
    # Count the total number of files in the directory tree, init tqdm
    total = sum(len(files) for _, _, files in os.walk(mask_dir)) + 1
    pbar = tqdm.tqdm(desc="Processing data", total=total)

    for root, _, files in os.walk(mask_dir):
        for file in files:
            # Update pbar
            pbar.update(1)

            if "eye_g" not in file:
                # Ignore no-glasses
                continue
            
            # Get the train/val/test type
            idx = int(file.split('_')[0])
            parent_path = split_info[idx]

            # Create the full path to original files
            mask_path = os.path.join(root, file)
            image_path = os.path.join(img_dir, str(idx) + ".jpg")

            # Create a save path of original files to train/val/test location
            image_save_path = os.path.join(parent_path, "images", str(idx) + ".jpg")
            mask_save_path = os.path.join(parent_path, "masks", file.replace(".png", ".jpg"))
            
            # Open the image, convert mask to black/white
            image = Image.open(image_path).resize(resize)
            mask = Image.open(mask_path).resize(resize)
            mask = Image.fromarray((np.array(mask) > 0).astype(np.uint8) * 255)

            # Save the mask and the image
            image.save(image_save_path)
            mask.save(mask_save_path)
    
    # Final update
    pbar.update(1)
    pbar.set_description("Done")

def main():
    # Perse the arguments
    args = parse_args()
    
    # Create train/val/test split info
    split_info = generate_split_paths(
        args.split_info_file_paths,
        args.celeba_mapping_file_path,
        args.save_dir
    )

    # Walk through samples and process
    walk_through_masks(
        args.mask_dir,
        args.img_dir,
        split_info,
        args.resize
    )

if __name__ == "__main__":
    main()