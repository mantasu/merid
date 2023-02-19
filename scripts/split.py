import os
import cv2
import shutil
import argparse
from tqdm import tqdm

CELEBA_DIR = "./data/celeba"
MEGLASS_DIR = "./data/meglass"
SYNTHETIC_DIR = "./data/synthetic"
LFW_DIR = "./data/lfw"

CELEBA_IMG_DIR = os.path.join(CELEBA_DIR, "aligned/align_size(256,256)_move(0.250,0.000)_face_factor(0.450)_jpg/data")
MEGLASS_IMG_DIR = os.path.join(MEGLASS_DIR, "MeGlass_ori")
SYNTHETIC_IMG_DIR = os.path.join(SYNTHETIC_DIR, "ALIGN_RESULT_v2")
LFW_IMG_DIR = os.path.join(LFW_DIR, "lfw-deepfunneled")

CELEBA_META_PATH = os.path.join(CELEBA_DIR, "list_attr_celeba.txt")
MEGLASS_META_PATH = os.path.join(MEGLASS_DIR, "meta.txt")
SYNTHETIC_META_PATH = os.path.join(SYNTHETIC_DIR, "basic_split.txt")
LFW_META_PATH = os.path.join(LFW_DIR, "lfw_attributes.txt")

AUTO_ARGS = {
    "celeba": {
        "img_dir": CELEBA_IMG_DIR,
        "meta_path": CELEBA_META_PATH,
        "out_dir_x": os.path.join(CELEBA_DIR, "train_x"),
        "out_dir_y": os.path.join(CELEBA_DIR, "train_y")
    },
    "synthetic": {
        "img_dir": SYNTHETIC_IMG_DIR,
        "meta_path": SYNTHETIC_META_PATH,
        "out_dir_x": os.path.join(SYNTHETIC_DIR, "train_x"),
        "out_dir_y": os.path.join(SYNTHETIC_DIR, "train_y")
    },
    "meglass": {
        "img_dir": MEGLASS_IMG_DIR,
        "meta_path": MEGLASS_META_PATH,
        "out_dir_x": os.path.join(MEGLASS_DIR, "test_x"),
        "out_dir_y": os.path.join(MEGLASS_DIR, "test_y")
    },
    "lfw": {
        "img_dir": LFW_IMG_DIR,
        "meta_path": LFW_META_PATH,
        "out_dir_x": os.path.join(LFW_DIR, "test_x"),
        "out_dir_y": os.path.join(LFW_DIR, "test_y")
    }
}

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="celeba", type=str, choices=["synthetic", "celeba", "meglass", "lfw"],
                        help="The type of dataset. Defaults to 'celeba'")
    parser.add_argument("--img_dir", dest="img_dir", type=str, default="auto",
                        help="The path to the folder with images. Defaults to 'auto'")
    parser.add_argument("--meta_path", type=str, default="auto",
                        help="The path to the dataset metadata. Defaults to 'auto'")
    parser.add_argument("--out_dir_x", type=str, default="auto",
                        help="The output folder to save training samples to. Defaults to 'auto'")
    parser.add_argument("--out_dir_y", type=str, default="auto",
                        help="The output folder to save training labels to. Defaults to 'auto'")
    parser.add_argument('--resize_w', type=int, default=-1,
                        help="The width to resize each image to. Defaults to -1 (no resize)")
    parser.add_argument('--resize_h', type=int, default=-1,
                        help="The height to resize each image to. Defaults to -1 (no resize)")

    return parser.parse_args()

def configure_arguments(args):
    if args.img_dir == "auto":
        args.img_dir = AUTO_ARGS[args.dataset]["img_dir"]
    
    if args.meta_path == "auto":
        args.meta_path = AUTO_ARGS[args.dataset]["meta_path"]
    
    if args.out_dir_x == "auto":
        args.out_dir_x = AUTO_ARGS[args.dataset]["out_dir_x"]
    
    if args.out_dir_y == "auto":
        args.out_dir_y = AUTO_ARGS[args.dataset]["out_dir_y"]
    
    return args

def process_line(line, args):
    # Split to attributes
    info = line.split('\t' if args.dataset == "lfw" else ' ')
    info = list(filter(lambda x: x != '', info))
    saved = False

    if args.dataset == "lfw":
        # Change some naming and indexing for LFW
        foldername = '_'.join(info[0].split())
        filename = f"{foldername}_{int(info[1]):04d}.jpg"
        info[0] = os.path.join(foldername, filename)

    # Generate input and output paths
    in_path = os.path.join(args.img_dir, info[0])

    if not os.path.isfile(in_path):
        # If the file is corrupted
        return 0, 1

    # Determine whether it is an input or a label and create corresponding path
    out_dir = args.out_dir_x if float(info[args.is_y_idx]) > 0 else args.out_dir_y
    out_path = os.path.join(out_dir, os.path.basename(info[0]))
                
    if args.resize_w != -1 or args.resize_h != -1:
        # Read the image to check dimensions for resize        
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)

        if img.shape[0] != args.resize_h or img.shape[1] != args.resize_w:
            # Set dimensions and resize to them
            dim = (args.resize_w, args.resize_h)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(out_path, img)
            saved = True
    
    if not saved:
        # Just copy over the file to new dir
        shutil.copyfile(in_path, out_path)
    
    return 1, 0

def split(args):
    # Creat the output directories
    os.makedirs(args.out_dir_x, exist_ok=True)
    os.makedirs(args.out_dir_y, exist_ok=True)

    with open(args.meta_path, 'r') as f:
        # Read lines, set is_glasses index
        meta = f.readlines()
        args.is_y_idx = 1
        num_success, num_fail = 0, 0

        if args.dataset == "celeba" or args.dataset == "lfw":
            # Celeba is unique
            meta = meta[2:]
            args.is_y_idx = 16 if args.dataset == "celeba" else 17

        for line in (pbar := tqdm(meta)):
            # Get if saved successfully or failed
            success, fail = process_line(line, args)
            num_success += success
            num_fail += fail

            # Update the progress bar description
            pbar.set_description(f"Successful: {num_success} | "
                                 f"Failed: {num_fail}")

if __name__ == "__main__":
    args = parse_arguments()
    args = configure_arguments(args)
    split(args)