import os
import sys
import tqdm
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as T

from PIL import Image

sys.path.append("data/ffhq/face-parsing.PyTorch-master")

from model import BiSeNet


def parse_args() -> argparse.Namespace:
    # Instantiate arguments parser
    parser = argparse.ArgumentParser()
    
    # Define all the command line arguments and default values
    parser.add_argument("--img-dir", type=str, default="data/ffhq/resized",
        help=f"Path to FFHQ images. Defaults to 'data/ffhq/resized'.")
    parser.add_argument("--model-path", type=str,
        default="data/ffhq/79999_iter.pth",
        help=f"Path to face parsing model. Defaults to "
             f"'data/ffhq/79999_iter.pth'.")
    parser.add_argument("--save-dir", type=str, default="data/ffhq",
        help=f"Path to save the processed data (test split). Defaults to "
             f"'data/ffqh'.")
    parser.add_argument("--device", type=str, default="cuda:0",
        help=f"The device to use for face attribute inference. Defaults to "
             f"'cuda:0'.")
    
    return parser.parse_args()

@torch.no_grad()
def check_if_glasses(image: Image.Image, model: nn.Module, device: str) -> bool:
    # Initialize transform
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # image = image.resize((512, 512), Image.BILINEAR)
    image = transform(image).unsqueeze(0).to(device)
    parse = model(image)[0].squeeze(0).argmax(0).unique()

    return 6 in parse

def process_files(img_dir: str, save_dir: str, **kwargs):
    glasses_dir = os.path.join(save_dir, "test", "glasses")
    no_glasses_dir = os.path.join(save_dir, "test", "no_glasses")

    os.makedirs(glasses_dir, exist_ok=True)
    os.makedirs(no_glasses_dir, exist_ok=True)

    for file in tqdm.tqdm(list(os.listdir(img_dir))):
        # Read the image form the original image path
        image = Image.open(os.path.join(img_dir, file))

        if check_if_glasses(image, **kwargs):
            # Set the path to test/glasses if detected
            save_path = os.path.join(glasses_dir, file)
        else:
            # Set the path to test/no_glasses if no glasses
            save_path = os.path.join(no_glasses_dir, file)
        
        # Save the image
        image.save(save_path)

def main():
    args = parse_args()
    model = BiSeNet(n_classes=19).to(args.device).eval()
    model.load_state_dict(torch.load(args.model_path))
    process_files(args.img_dir, args.save_dir, model=model, device=args.device)


if __name__ == "__main__":
    main()