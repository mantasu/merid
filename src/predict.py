import os
import torch
import argparse
from models.merid.merid import Merid
from utils.config import parse_config

def parse_args() -> argparse.Namespace:
    # Instantiate arguments parser
    parser = argparse.ArgumentParser()
    
    # Define all the command line arguments and default values
    parser.add_argument("-c", "--config", type=str,
        default="configs/pretrained.json",
        help=f"Path to `.json` config that specifies how the inference model "
             f"should be loaded. Defaults to 'configs/pretrained.json'.")
    parser.add_argument("-i", "--input", type=str, default="data/demo",
        help=f"Path to a single `.jpg` image or to a folder of `.jpg` images "
             f"to be used for prediction. Defaults to 'data/demo'.")
    
    return parser.parse_args()

def load_model(config: str) -> Merid:
    config = parse_config(config)
    model = Merid(config).to("cuda:0" if torch.cuda.is_available() else "cpu")

    return model

def load_image_paths(path: str) -> str | list[str]:
    if os.path.isfile(path):
        return [path]
    
    return [os.path.join(path, file) for file in os.listdir(path)]

def main():
    args = parse_args()
    paths = load_image_paths(args.input)
    model = load_model(args.config)

    images = model.predict(paths)

    if not isinstance(images, list):
        images = [images]

    for image, in_path in zip(images, paths):
        image.save(in_path[:-4] + "_pred.jpg")

if __name__ == "__main__":
    main()
    # from models.nafnet.nafnet import NAFNet
    # from models.merid.recolorizer import Recolorizer
    # from collections import OrderedDict
    
    # new_weights = OrderedDict()
    # weights = torch.load("checkpoints/recolorizer-best.pth")

    # for key in weights.keys():
    #     new_weights[key[12:]] = weights[key]

    # nafnet = Recolorizer()
    # nafnet.load_state_dict(new_weights)
    # torch.save(nafnet.state_dict(), "recolorizer-pure.pth")
