import os
import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms as T

from typing import Sequence
from torch.utils.data import Dataset

class GlassesLabeledDataset(Dataset):
    def __init__(self,
                 path_x: str | os.PathLike,
                 path_y: str | os.PathLike | None = None,
                 replacable_ending: str = "-all",
                 label_identifiers: Sequence[tuple[str, bool]] = [],
                 transform: A.Compose | T.Compose | None = None,
                 load_in_memory: bool = False
                ):
        super().__init__()

        # Init some basic dataset attributes
        self.load_in_memory = load_in_memory
        self.samples = []

        # Specify the transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((.5, .5, .5), (.5, .5, .5))
        ]) if transform is None else transform
        
        for filename in os.listdir(path_x):
            # Initialize the sample as dict and init counters
            sample = {"image": os.path.join(path_x, filename)}
            image_count, mask_count = 1, 0

            for identifier in label_identifiers:
                # Create label filename based on identifier, join with root
                _filename = filename.replace(replacable_ending, identifier[0])
                path = os.path.join(path_y, _filename)

                if identifier[1]:
                    # Create a key for label (mask) name and assign path
                    key = "mask" + (str(mask_count) if mask_count > 0 else "")
                    sample[key] = path
                    mask_count += 1
                else:
                    # Create a key for label (image) name and assign path
                    sample[f"image{image_count}"] = path
                    image_count += 1

            if load_in_memory:
                # Load the whole tensors into memory
                sample = self.paths_to_tensors(sample)
            
            # Append the sample to samples
            self.samples.append(sample)
    
    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor]:
        # Get the specified sample
        sample = self.samples[index]

        if not self.load_in_memory:
            # If not already loaded in memory
            sample = self.paths_to_tensors(sample)
        
        # Convert dictionary to tuple
        sample = tuple(sample.values())
        
        return sample[0] if len(sample) == 1 else sample

    def __len__(self) -> int:
        return len(self.samples)
    
    def paths_to_tensors(self, sample: dict[str, str | os.PathLike]) -> \
                         dict[str, torch.Tensor]:
        # Init transforms
        transforms = {}
        
        for name, path in sample.items():
            if name.startswith("image"):
                # If normal 3-color image, just convert from BGR to RGB
                image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            else:
                # If grayscale, read as (H, W), ignore channel
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            # Assign image to transforms
            transforms[name] = image
        
        if isinstance(self.transform, A.Compose):
            # A.Compose expects named arguments
            transforms = self.transform(**transforms)
        else:
            # Stack all images on axis 0 to perform a transform
            aug = self.transform(np.stack(
                np.tile(img, (1, 1, 3)) if img.ndim == 2 else img
                for img in transforms.values()
            ))

            # Name samples/labels, take only one channel for grayscale
            transforms = {
                k: aug[i, 0, 0] if k.startswith("mask") else aug[i]
                for i, k in enumerate(transforms.keys())
            }
        
        return transforms
            