import os
import cv2
import dlib
import tqdm
import argparse
import numpy as np

def parse_args() -> argparse.Namespace:
    # Instantiate arguments parser
    parser = argparse.ArgumentParser()
    
    # Define all the command line arguments and default values
    parser.add_argument("--img-dir", type=str, default="data/lfw/lfw",
        help=f"Path to LFW images. Defaults to 'data/lfw/lfw'.")
    parser.add_argument("--save-dir", type=str, default="data/lfw",
        help=f"Path to save the processed data (test split). Defaults to "
             f"'data/lfw'.")
    parser.add_argument("--attributes-path",
        default="data/lfw/lfw_attributes.txt",
        help=f"Path to `.txt` file containing image attributes. Defaults to "
             f"'data/lfw/lfw_attributes.txt'.")
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

def align_save_pair(
    input_path1: str,
    input_path2: str,
    output_path1: str,
    output_path2: str,
    **kwargs,
):
    # Read the image as numpy of shape (H, W, C)
    image1  = cv2.imread(input_path1)
    image2  = cv2.imread(input_path2)

    # Estimate face1 landmarks
    landmarks1 = get_face_landmarks(
        image=image1,
        detector=kwargs["detector"],
        predictor=kwargs["predictor"]
    )

    # Estimate face2 landmarks
    landmarks2 = get_face_landmarks(
        image=image1,
        detector=kwargs["detector"],
        predictor=kwargs["predictor"]
    )

    if landmarks1 is None or landmarks2 is None:
        return

    # Use landmarks to align face1
    image_aligned1 = align_crop(
        image=image1,
        landmarks_src=landmarks1,
        landmarks_standard=kwargs["landmarks_standard"],
        face_factor=kwargs["face_factor"],
        crop_size=kwargs["crop_size"]
    )

    # Use landmarks to align face2
    image_aligned2 = align_crop(
        image=image2,
        landmarks_src=landmarks2,
        landmarks_standard=kwargs["landmarks_standard"],
        face_factor=kwargs["face_factor"],
        crop_size=kwargs["crop_size"]
    )
    
    # Set up the image quality and save the images
    quality = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    cv2.imwrite(output_path1, image_aligned1, params=quality)
    cv2.imwrite(output_path2, image_aligned2, params=quality)

def process_dirs(
    img_dir: str,
    save_dir: str,
    certainly_is_glasses: dict[str, bool],
    certainly_no_glasses: dict[str, bool],
    **kwargs,
):
    # Create a glasses root directory and non-glasses root dir
    glasses_dir = os.path.join(save_dir, "test", "glasses")
    no_glasses_dir = os.path.join(save_dir, "test", "no_glasses")

    # Actually create the folders that will contain data
    os.makedirs(glasses_dir, exist_ok=True)
    os.makedirs(no_glasses_dir, exist_ok=True)

    for identity in tqdm.tqdm(list(os.listdir(img_dir))):
        # Read the image directory to create a list of images in it
        image_files = list(os.listdir(os.path.join(img_dir, identity)))

        if len(image_files) < 2:
            # Return if just 1/0
            continue
        
        # Initialize glasses/no-glasses
        glasses, no_glasses = [], []

        for image_file in image_files:
            if certainly_is_glasses.get(image_file, False):
                # If certainly is glasses
                glasses.append(image_file)
            elif certainly_no_glasses.get(image_file, False):
                # If certainly no glasses
                no_glasses.append(image_file)
        
        for i, (_glasses, _no_glasses) in enumerate(zip(glasses, no_glasses)):
            # Create a naming function for output file
            name = lambda x: x[:-4] + f"_pair_{i+1}.jpg"
            
            # Create a source path by joining root and image name
            source_glasses = os.path.join(img_dir, identity, _glasses)
            target_glasses = os.path.join(glasses_dir, name(_glasses))

            # Create a target path by joining save root dir and img name
            source_no_glasses = os.path.join(img_dir, identity, _no_glasses)
            target_no_glasses = os.path.join(no_glasses_dir, name(_no_glasses))

            # Align a pair of images
            align_save_pair(
                source_glasses,
                source_no_glasses,
                target_glasses,
                target_no_glasses,
                **kwargs
            )

def parse_attributes(attributes_path: str):
    # Initialize a list of certainly glasses and not
    certainly_is_glasses, certainly_no_glasses = {}, {}

    with open(attributes_path, 'r') as f:
        for line in f.readlines()[2:]:
            # Read every line in the `.txt` file, split to attributes
            info = list(filter(lambda x: x != '', line.split('\t')))
            filename = f"{'_'.join(info[0].split())}_{int(info[1]):04d}.jpg"

            # Create certainty boundaries that tell whether is glasses
            is_glasses = float(info[15]) < -1.3 or float(info[16]) > 0.3 or \
                         float(info[17]) > 0
            no_glasses = float(info[15]) > 0.1 and not is_glasses

            # Assign certainty labels to mapping dict
            certainly_is_glasses[filename] = is_glasses
            certainly_no_glasses[filename] = no_glasses
    
    return certainly_is_glasses, certainly_no_glasses

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

    # Parse the certainty labels to guide which iage goes where
    certainty = parse_attributes(kwargs.pop("attributes_path"))
    kwargs["certainly_is_glasses"] = certainty[0]
    kwargs["certainly_no_glasses"] = certainty[1]

    # Process all the directories
    process_dirs(**kwargs)

if __name__ == "__main__":
    main()
