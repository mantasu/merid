import os
import sys
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl

sys.path.append("src")

from base_dataset import BaseDataset
from utils.io import load_image

class SyntheticDataset(BaseDataset):
    def __init__(self,
                 data_path: str | os.PathLike,
                 target: str = "train",
                 transform: A.Compose | None = None,
                 seed: int | None = None
                ):
        super().__init__(data_path, target, transform, seed)

        # Reconfigure default transform
        self.transform = A.Compose(
            [A.Normalize(), ToTensorV2()], 
            additional_targets={"image0": "image"}
        ) if transform is None else transform
    
    def init_samples(self, data_path: str) -> list[tuple[str, str, str, str]]:
        samples = []

        for file in os.listdir(os.path.join(data_path, "glasses")):
            samples.append((
                os.path.join(data_path, "glasses", file),
                os.path.join(data_path, "no_glasses", file.replace("all", "face")),
                os.path.join(data_path, "masks", file.replace("all", "seg")),
                os.path.join(data_path, "masks", file.replace("all", "shseg"))
            ))
        
        return samples

    def load_sample(self, glasses_path: str, no_glasses_path: str,
                    mask_glasses_path: str, mask_shadows_path: str) \
                    -> tuple[torch.Tensor]:
        
        # Load images to a list of images
        images = [load_image(glasses_path),
                  load_image(no_glasses_path)]
        
        # Load masks to a list of masks and convert to grayscale
        masks = [load_image(mask_glasses_path, convert_to_grayscale=True),
                 load_image(mask_shadows_path, convert_to_grayscale=True)]

        aug = self.transform(images=images, masks=masks)
        aug["masks"] = [(mask > 0).float() for mask in aug["masks"]]
        
        return (*aug["images"], *aug["masks"])


class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_path: str | os.PathLike = "data/synthetic",
                 augment_train: bool = True,
                 **loader_kwargs):
        super().__init__(SyntheticDataset, data_path, augment_train, **loader_kwargs)
