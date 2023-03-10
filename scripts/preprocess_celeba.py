import os
import re
import cv2
import tqdm
import argparse
import numpy as np

from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

def parse_args() -> argparse.Namespace:
    # Instantiate arguments parser
    parser = argparse.ArgumentParser()
    
    # Define all the command line arguments and default values
    parser.add_argument("--img-dir", default="data/celeba/img_celeba",
        help="Path to CelebA images. Defaults to 'data/celeba/img_celeba'.")
    parser.add_argument("--landmarks_path", default="data/celeba/landmark.txt",
        help=f"Path to CelebA landmarks. Defaults to "
             f"'data/celeba/landmark.txt'.")
    parser.add_argument("--landmarks_standard_path",
        default="data/celeba/standard_landmark_68pts.txt",
        help=f"Path to CelebA standard landmaks. Defaults to "
             f"'data/celeba/standard_landmark_68pts.txt'.")
    parser.add_argument("--split-info-file-paths", nargs=3, default=\
        [f"data/celeba/{x}_label.txt" for x in ["train", "val", "test"]],
        help=f"3 paths to files that contain information on how to split the "
             f"data into 3 parts. Defaults to ['data/celeba/train_label.txt', "
             f"'data/celeba/val_label.txt', 'data/celeba/test_label.txt'].")
    parser.add_argument("--crop-size", nargs=2, type=int, default=[256, 256],
        help=f"The size to crop and resize the faces to (width and height). "
             f"Defaults to [256, 256].")
    parser.add_argument("--face-factor", type=float, default=0.65,
        help=f"The factor of face area relative to the output image. "
             f"Defaults to 0.65.")
    
    return vars(parser.parse_args())

def align_crop(image: np.ndarray,
               landmarks_src: np.ndarray,
               landmarks_standard: np.ndarray,
               face_factor: float = 0.7,
               crop_size: tuple[int, int] = (256, 256)
               ) -> np.ndarray:
    # Compute target landmarks based on the provided face factor
    target_landmarks = landmarks_standard * max(*crop_size) * face_factor
    target_landmarks += np.array([crop_size[0] // 2, crop_size[1] // 2])
    
    # Estimate transform matrix based on similarity for alignment
    transform = cv2.estimateAffinePartial2D(target_landmarks, landmarks_src,
                                            ransacReprojThreshold=np.Inf)[0]
    
    # Acquire the cropped image based on the estimated transform
    image_cropped = cv2.warpAffine(image, transform, crop_size,
                                   flags=cv2.WARP_INVERSE_MAP + cv2.INTER_AREA,
                                   borderMode=cv2.BORDER_REPLICATE)

    return image_cropped

def open_landmark_files(landmarks_path: str,
                        landmarks_standard_path: str
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(landmarks_path) as f:
        # Read the line
        line = f.readline()
    
    # Split line to attributes, and create lanmarks range
    num_landmarks = len(re.split("[ ]+", line)[1:]) // 2
    landmarks_range = range(1, num_landmarks * 2 + 1)

    # Read image filenames, landmarks for images and a standard landmarks file
    image_filenames = np.genfromtxt(landmarks_path, dtype=str, usecols=0)
    landmarks = np.genfromtxt(landmarks_path, dtype=float, usecols=landmarks_range)
    landmarks_standard = np.genfromtxt(landmarks_standard_path, dtype=float)

    # Reshape to (N, 68, 2), move standard ones slightly
    landmarks = landmarks.reshape(-1, num_landmarks, 2)
    landmarks_standard = landmarks_standard.reshape(num_landmarks, 2)
    landmarks_standard[:, 1] += 0.25

    return image_filenames, landmarks, landmarks_standard

def generate_save_paths(split_info_file_paths: list[str, str, str],
                        img_dir: str) -> dict[str, tuple[str, str]]:
    # Check parent data dir, init paths map
    parent_dir = os.path.dirname(img_dir)
    save_paths_map = {}
    
    for split_info_path in split_info_file_paths:
        # Determine the type of split
        if "train" in split_info_path:
            split_type = "train"
        elif "val" in split_info_path:
            split_type = "val"
        elif "test" in split_info_path:
            split_type = "test"
        
        with open(split_info_path, 'r') as f:
            for line in f.readlines():
                # Read line info
                info = line.split()

                # Generate input and output image paths
                in_path = os.path.join(img_dir, info[0])

                if not os.path.isfile(in_path):
                    # If the file is corrupted
                    continue
                
                # Determine the label, finalize output path, add to map
                dir_name = "glasses" if int(info[16]) > 0 else "no_glasses"
                out_path = os.path.join(parent_dir, split_type, dir_name, info[0])
                save_paths_map[info[0]] = in_path, out_path

                if not os.path.exists(dirs := os.path.dirname(out_path)):
                    # Ensure folders exist
                    os.makedirs(dirs)
    
    return save_paths_map

def worker(save_paths_map: dict[str, tuple[str, str]],
           image_filenames: np.ndarray,
           landmarks: np.ndarray,
           landmarks_standard: np.ndarray,
           i: int,
           **kwargs):
    # Retrieve the image input and output paths, read image
    in_path, out_path = save_paths_map[image_filenames[i]]
    image = cv2.imread(in_path)

    # Align and crop the image and save to a specified/labeled path 
    image_aligned = align_crop(image, landmarks[i], landmarks_standard, **kwargs)
    cv2.imwrite(out_path, image_aligned, params=[int(cv2.IMWRITE_JPEG_QUALITY), 95])

def main():
    # Parse arguments
    kwargs = parse_args()

    # Pop the dir/path arguments
    img_dir = kwargs.pop("img_dir")
    landmarks_path = kwargs.pop("landmarks_path")
    split_info_file_paths = kwargs.pop("split_info_file_paths")
    landmarks_standard_path = kwargs.pop("landmarks_standard_path")

    # Generate in/out paths and create alignment transforms
    save_paths_map = generate_save_paths(split_info_file_paths, img_dir)
    out = open_landmark_files(landmarks_path, landmarks_standard_path)
    worker_fn = partial(worker, save_paths_map, *out, **kwargs)

    with ThreadPool(cpu_count()) as pool:
        # Generate a pool map, pass it to progress bar and execute
        pool_map = pool.imap_unordered(worker_fn, range(len(out[0])))
        list(tqdm.tqdm(pool_map, total=len(out[0])))

if __name__ == "__main__":
    main()
