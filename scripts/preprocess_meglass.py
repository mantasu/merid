import os
import cv2
import tqdm
import dlib
import argparse
import numpy as np

from functools import partial
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count, Manager, Lock


# Global lock
LOCK = Lock()

def parse_args() -> argparse.Namespace:
    # Instantiate arguments parser
    parser = argparse.ArgumentParser()
    
    # Define all the command line arguments and default values
    parser.add_argument("--img-dir", type=str,
        default="data/meglass/MeGlass_ori",
        help=f"Path to MeGlass images. Defaults to "
             f"'data/meglass/MeGlass_ori'.")
    parser.add_argument("--gallery-black-glass-path", type=str,
        default="data/meglass/gallery_black_glass.txt",
        help=f"Path to `.txt` file which contains image names for gallery "
             f"glasses. Defaults to 'data/meglass/gallery_black_glass.txt'.")
    parser.add_argument("--gallery-no-glass-path", type=str,
        default="data/meglass/gallery_no_glass.txt",
        help=f"Path to `.txt` file which contains image names for gallery no "
             f"glasses. Defaults to 'data/meglass/gallery_no_glass.txt'.")
    parser.add_argument("--probe-black-glass-path", type=str,
        default="data/meglass/probe_black_glass.txt",
        help=f"Path to `.txt` file which contains image names for probe "
             f"glasses. Defaults to 'data/meglass/probe_black_glass.txt'.")
    parser.add_argument("--probe-no-glass-path", type=str,
        default="data/meglass/probe_no_glass.txt",
        help=f"Path to `.txt` file which contains image names for probe no "
             f"glasses. Defaults to 'data/meglass/probe_no_glass.txt'.")
    parser.add_argument("--save-dir", type=str, default="data/meglass",
        help=f"Path to save the processed data (test split). Defaults to "
             f"'data/meglass'.")
    parser.add_argument("--landmarks-standard-path",
        default="data/celeba/standard_landmark_68pts.txt",
        help=f"Path to `.txt` file listing standard landmarks. Defaults to "
             f"'data/celeba/standard_landmark_68pts.txt'.")
    parser.add_argument("--landmarks-predictor-path", type=str,
        default="data/lfw/shape_predictor_68_face_landmarks.dat",
        help=f"Path to landmarks shape predictor `.dat` file used by dlib. "
             f"Defaults to 'data/lfw/shape_predictor_68_face_landmarks.dat'.")
    parser.add_argument("--split-frac", nargs=2, type=float, default=[.1, .1],
        help=f"Two float numbers determining the validation and the test data "
             f"fractions of the whole dataset. Defaults to [0.1, 0.1].")
    parser.add_argument("--crop-size", nargs=2, type=int, default=[256, 256],
        help=f"The size to crop and resize the faces to (width and height). "
             f"Defaults to [256, 256].")
    parser.add_argument("--face-factor", type=float, default=0.65,
        help=f"The factor of face area relative to the output image. "
             f"Defaults to 0.65.")
    
    return vars(parser.parse_args())


def align_crop(
    image: np.ndarray,
    landmarks_src: np.ndarray,
    landmarks_standard: np.ndarray,
    face_factor: float = 0.65,
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

def get_landmarks_tools(
    landmarks_standard_path: str,
    landmarks_predictor_path: str
) -> tuple[np.ndarray, dlib.fhog_object_detector, dlib.shape_predictor]:
    # Read the standard landmarks coordinates from the specified file
    landmarks_standard = np.genfromtxt(landmarks_standard_path, dtype=float)
    landmarks_standard = landmarks_standard.reshape(-1, 2)
    landmarks_standard[:, 1] += 0.25

    # Initialize a dlib face detector and face landmarks predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmarks_predictor_path)

    return landmarks_standard, detector, predictor

def get_face_landmarks(
    image: np.ndarray,
    detector: dlib.fhog_object_detector,
    predictor: dlib.shape_predictor,
) -> np.ndarray:
    
    with LOCK:
        # Detect the faces
        faces = detector(image)

    if len(faces) == 0:
        # No faces
        return None
    
    # Predict landmarks for the first face
    landmarks = predictor(image, faces[0])

    # Convert the landmarks to a list of coordinates of x and y
    landmarks = [[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)]
        
    return np.array(landmarks, dtype=float)

def worker(save_dir, suffix, input_file, **kwargs):
    # Generate input and output paths
    output_file = input_file[:-4] + suffix
    input_path = os.path.join(kwargs["img_dir"], input_file)
    output_path = os.path.join(save_dir, output_file)

    # Read the image as numpy of shape (H, W, C)
    image  = cv2.imread(input_path)

    # Estimate face landmarks
    landmarks = get_face_landmarks(
        image=image,
        detector=kwargs["detector"],
        predictor=kwargs["predictor"]
    )

    if landmarks is None:
        return

    # Use landmarks to align face
    image_aligned = align_crop(
        image=image,
        landmarks_src=landmarks,
        landmarks_standard=kwargs["landmarks_standard"],
        face_factor=kwargs["face_factor"],
        crop_size=kwargs["crop_size"]
    )
    
    # Set up the image quality and save the image
    quality = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    cv2.imwrite(output_path, image_aligned, params=quality)

    # Add identity to path
    identity = '@'.join(input_file.split('@')[:2])
    file_list = kwargs["identities"].get(identity, [])
    kwargs["identities"][identity] = [*file_list, output_path]

def process_single_test_list(
    list_path: str,
    save_dir: str,
    suffix: str = ".jpg",
    **kwargs,
) -> set[str]:

    with open(list_path, 'r') as f:
        # Generate a list of filenames to loop through
        filenames = [line.strip() for line in f.readlines()]
    
    # Initialize a helper multithread worker function
    worker_fn = partial(worker, save_dir, suffix, **kwargs)

    with ThreadPool(cpu_count()) as pool:
        # Generate a pool map, pass to progress bar, execute
        pool_map = pool.imap_unordered(worker_fn, filenames)
        description = f"Processing test list [{kwargs['i']} / 4]"
        list(tqdm.tqdm(pool_map, desc=description, total=len(filenames)))


def process_test_lists(**kwargs):
    # Set up the directories to save files to
    save_dir = kwargs.pop("save_dir")
    glasses_test_dir = os.path.join(save_dir, "test", "glasses")
    no_glasses_test_dir = os.path.join(save_dir, "test", "no_glasses")
    
    # Create the directories where images will be saved
    os.makedirs(glasses_test_dir, exist_ok=True)
    os.makedirs(no_glasses_test_dir, exist_ok=True)

    # Create the lists to parse
    test_lists_to_parse = [
        (kwargs["gallery_black_glass_path"], glasses_test_dir, "_gallery.jpg"),
        (kwargs["gallery_no_glass_path"], no_glasses_test_dir, "_gallery.jpg"),
        (kwargs["probe_black_glass_path"], glasses_test_dir, "_probe.jpg"),
        (kwargs["probe_no_glass_path"], no_glasses_test_dir, "_probe.jpg"),
    ]

    # Create a prograss bar counter and shared identities dict
    kwargs.update({"i": 1, "identities": Manager().dict()})

    for list_path, save_dir, suffix in test_lists_to_parse:
        # For every `.txt` list file, apply multithreaded processing
        process_single_test_list(list_path, save_dir, suffix, **kwargs)
        kwargs["i"] += 1
    
    for key, val in kwargs["identities"].items():
        if len(val) == 4:
            # All fine
            continue

        for file_path in val:
            # Remove incomplete
            os.remove(file_path)
        
        # Remove incomplete identity
        del kwargs["identities"][key]
    
    # Print how many total identities were parsed successfully
    print(f"Processed {len(kwargs['identities'])} out of 1710.")

def main():
    # Parse the arguments
    kwargs = parse_args()
    
    # Parse standard landmarks, create detector and landmark predictor
    landmarks_standard, detector, predictor = get_landmarks_tools(
        kwargs["landmarks_standard_path"],
        kwargs["landmarks_predictor_path"],
    )

    # Assign the arguments to the kwargs dictionary
    kwargs["landmarks_standard"] = landmarks_standard
    kwargs["detector"] = detector
    kwargs["predictor"] = predictor

    # Pass kwargs to process all `.txt` list files
    process_test_lists(**kwargs)


if __name__ == "__main__":
    main()
