import os
import sys
import torch
import albumentations as A
import pytorch_lightning as pl

from .base_data import BaseDataset
from torch.utils.data import DataLoader

sys.path.append("src")
from utils.image_tools import load_image
from utils.training import create_augmentation

class CelebaMaksHQDataset(BaseDataset):
    def __init__(self,
                 data_path: str | os.PathLike = "data/celeba-mask-hq",
                 target: str ="train",
                 transform: A.Compose | None = None,
                 seed: int | None = None,
                ):
        super().__init__(data_path, target, transform, seed)

    def init_samples(self, data_path: str) -> list[tuple[str, str]]:
        samples = []

        for filename in os.listdir(os.path.join(data_path, "masks")):
            # Get the index of the sample
            idx = int(filename.split('_')[0])

            # Generate image path and mask path from teh specified target
            mask_path = os.path.join(data_path, "masks", filename)
            image_path = os.path.join(data_path, "images", str(idx) + ".jpg")

            # Append both paths as a tuple to sample array
            samples.append((image_path, mask_path))

        return samples

    def load_sample(self, image_path: str, mask_path: str) \
                    -> tuple[torch.Tensor, torch.Tensor]:
        # Read image, convert to grayscale if mask
        image = load_image(image_path)
        mask = load_image(mask_path, convert_to_grayscale=True)

        # Apply transformations to mask and image
        aug = self.transform(image=image, mask=mask)
        image, mask = aug["image"], aug["mask"]

        return image, (mask > 0).float()


class CelebaMaksHQModule(pl.LightningDataModule):
    def __init__(self,
                 data_path: str | os.PathLike = "data/celeba-mask-hq",
                 augment_train: bool = True,
                 **loader_kwargs
                ):
        super().__init__()
        # Assign dataset attributes
        self.data_path = data_path
        self.train_transform = None
        self.loader_kwargs = loader_kwargs

        if augment_train:
            self.train_transform = create_augmentation()

        # Set some default data loader arguments
        self.loader_kwargs.setdefault("batch_size", 10)
        self.loader_kwargs.setdefault("num_workers", 24)
        self.loader_kwargs.setdefault("pin_memory", True)

    def train_dataloader(self) -> DataLoader:
        # Create train dataset and return loader
        train_dataset = CelebaMaksHQDataset(
            data_path=self.data_path,
            target="train",
            transform=self.train_transform
        )
        return DataLoader(train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        # Create val dataset and return loader
        val_dataset = CelebaMaksHQDataset(
            data_path=self.data_path,
            target="val",
            seed=0
        )
        return DataLoader(val_dataset, **self.loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        # Create test dataset and return loader
        test_dataset = CelebaMaksHQDataset(
            data_path=self.data_path,
            target="test",
            seed=0
        )
        return DataLoader(test_dataset, **self.loader_kwargs)
