import os
import re
import cv2
import tqdm
import torch
import torchsr
import argparse
import numpy as np

from multiprocessing import Pool
from torchvision.transforms.functional import to_tensor, to_pil_image

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # main
    parser.add_argument("--img_dir", default="data/celeba/img_celeba")
    parser.add_argument("--landmarks_path", default="data/celeba/landmark.txt")
    parser.add_argument("--landmarks_standard_path", default=\
        "data/celeba/standard_landmark_68pts.txt")
    parser.add_argument("--split_info_file_paths", nargs=3, default=\
        [f"data/celeba/{x}_label.txt" for x in ["train", "val", "test"]])
    parser.add_argument("--crop-size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--face-factor", type=float, default=0.65, help="The factor of face area relative to the output image.")
    parser.add_argument("--sr-scale", type=int, default=1, choices=[1, 2, 3, 4, 8],
        help=f"By how much to increase the image quality (super resolution "
             f"fraction). Defaults to 1.")
    parser.add_argument("--sr-model", type=str, default="ninasr_b1",
        help=f"The super resolution model to use. For available options, see "
             f"https://pypi.org/project/torchsr/. Defaults to 'ninasr_b1'.")
    
    return vars(parser.parse_args())

def align_crop(image: np.ndarray, landmarks_src: np.ndarray,
               landmarks_standard: np.ndarray, face_factor: float = 0.7,
               crop_size: tuple[int, int] = (256, 256)):
    # set OpenCV
    # inter = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_AREA,
    #          3: cv2.INTER_CUBIC, 4: cv2.INTER_LANCZOS4, 5: cv2.INTER_LANCZOS4}
    # border = {'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE,
    #           'symmetric': cv2.BORDER_REFLECT, 'reflect': cv2.BORDER_REFLECT101,
    #           'wrap': cv2.BORDER_WRAP}

    # # check
    # assert align_type in ['affine', 'similarity'], 'Invalid `align_type`! Allowed: %s!' % ['affine', 'similarity']
    # assert order in [0, 1, 2, 3, 4, 5], 'Invalid `order`! Allowed: %s!' % [0, 1, 2, 3, 4, 5]
    # assert mode in ['constant', 'edge', 'symmetric', 'reflect', 'wrap'], 'Invalid `mode`! Allowed: %s!' % ['constant', 'edge', 'symmetric', 'reflect', 'wrap']

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

    # get transformed landmarks
    # tformed_landmarks = cv2.transform(np.expand_dims(src_landmarks, axis=0), cv2.invertAffineTransform(tform))[0]

    return image_cropped


# ==============================================================================
# =                                opencv first                                =
# ==============================================================================

# _DEAFAULT_JPG_QUALITY = 95
# import cv2
# imread = cv2.imread
# imwrite = partial(cv2.imwrite, params=[int(cv2.IMWRITE_JPEG_QUALITY), _DEAFAULT_JPG_QUALITY])
# align_crop = align_crop_opencv

# ==============================================================================
# =                                     run                                    =
# ==============================================================================

def open_landmark_files(landmarks_path: str, landmarks_standard_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(landmarks_path) as f:
        line = f.readline()
    
    num_landmarks = len(re.split("[ ]+", line)[1:]) // 2
    landmarks_range = range(1, num_landmarks * 2 + 1)

    image_filenames = np.genfromtxt(landmarks_path, dtype=str, usecols=0)
    landmarks = np.genfromtxt(landmarks_path, dtype=float, usecols=landmarks_range)
    landmarks_standard = np.genfromtxt(landmarks_standard_path, dtype=float)

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



# count landmarks
# with open(args.landmark_file) as f:
#     line = f.readline()
# n_landmark = len(re.split('[ ]+', line)[1:]) // 2

# # read data

# standard_landmark[:, 0] += args.move_w
# standard_landmark[:, 1] += args.move_h

# # data dir
# save_dir = os.path.join(args.save_dir, 'align_size(%d,%d)_move(%.3f,%.3f)_face_factor(%.3f)_%s' % (args.crop_size_h, args.crop_size_w, args.move_h, args.move_w, args.face_factor, args.save_format))
# data_dir = os.path.join(save_dir, 'data')
# if not os.path.isdir(data_dir):
#     os.makedirs(data_dir)

def worker(i: int, save_paths_map: dict[str, tuple[str, str]],
           image_filenames: np.ndarray, landmarks: np.ndarray,
           landmarks_standard: np.ndarray, sr_model: torch.nn.Module | None,
           **kwargs):
    # Retrieve the image input and output paths, read image
    in_path, out_path = save_paths_map[image_filenames[i]]
    image = cv2.cvtColor(cv2.imread(in_path), cv2.COLOR_BGR2RGB)

    # Align and crop the image and save to a specified/labeled path 
    image_aligned = align_crop(image, landmarks[i], landmarks_standard, **kwargs)

    if sr_model is not None:
        device = next(sr_model.parameters()).device
        sr_input = to_tensor(image_aligned).unsqueeze(0).to(device)
        image_aligned = sr_model(sr_input).clip(0, 1).squeeze()
        image_aligned = np.array(to_pil_image(image_aligned))
        image_aligned = cv2.resize(image_aligned, kwargs["crop_size"], interpolation=cv2.INTER_AREA)
    
    image_aligned = cv2.cvtColor(image_aligned, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, image_aligned, params=[int(cv2.IMWRITE_JPEG_QUALITY), 95])

    

# def work(i):  # a single work
#     for _ in range(3):  # try three times
#         try:
#             img = imread(os.path.join(args.img_dir, img_names[i]))
#             img_crop, tformed_landmarks = align_crop(img,
#                                                      landmarks[i],
#                                                      standard_landmark,
#                                                      crop_size=(args.crop_size_h, args.crop_size_w),
#                                                      face_factor=args.face_factor,
#                                                      align_type=args.align_type,
#                                                      order=args.order,
#                                                      mode=args.mode)

#             name = os.path.splitext(img_names[i])[0] + '.' + args.save_format
#             path = os.path.join(data_dir, name)
#             if not os.path.isdir(os.path.split(path)[0]):
#                 os.makedirs(os.path.split(path)[0])
#             imwrite(path, img_crop)

#             tformed_landmarks.shape = -1
#             name_landmark_str = ('%s' + ' %.1f' * n_landmark * 2) % ((name, ) + tuple(tformed_landmarks))
#             succeed = True
#             break
#         except:
#             succeed = False
#     if succeed:
#         return name_landmark_str
#     else:
#         print('%s fails!' % img_names[i])


# pool = Pool(args.n_worker)
# name_landmark_strs = list(tqdm.tqdm(pool.imap(work, range(len(img_names))), total=len(img_names)))
# pool.close()
# pool.join()

# landmarks_path = os.path.join(save_dir, 'landmark.txt')
# with open(landmarks_path, 'w') as f:
#     for name_landmark_str in name_landmark_strs:
#         if name_landmark_str:
#             f.write(name_landmark_str + '\n')

def main():
    # Parse arguments
    kwargs = parse_args()
    sr_scale = kwargs.pop("sr_scale")
    model_fn = getattr(torchsr.models, kwargs.pop("sr_model"))

    # Pop the dir/path arguments
    img_dir = kwargs.pop("img_dir")
    landmarks_path = kwargs.pop("landmarks_path")
    split_info_file_paths = kwargs.pop("split_info_file_paths")
    landmarks_standard_path = kwargs.pop("landmarks_standard_path")

    if sr_scale > 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sr_model = model_fn(sr_scale, pretrained=True).to(device).eval()

        for param in sr_model.parameters():
            param.requirese_grad = False
    else:
        sr_model = None


    # Generate in/out paths and create alignment transforms
    save_paths_map = generate_save_paths(split_info_file_paths, img_dir)
    out = open_landmark_files(landmarks_path, landmarks_standard_path)
    worker_fn = lambda x: worker(x, save_paths_map, *out, sr_model, **kwargs)

    kwargs.pop("num_workers")

    for i in tqdm.tqdm(range(len(out[0]))):
        worker_fn(i)

    # pool = Pool(kwargs.pop("num_workers"))
    # tqdm.tqdm(pool.imap(worker_fn, range(len(out[0]))), total=len(out[0]))
    # pool.close()
    # pool.join()

if __name__ == "__main__":
    main()