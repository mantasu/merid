import os
import sys
import torch
import random
import numpy as np
import albumentations as A
import pytorch_lightning as pl

from typing import Any, Iterable
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
from albumentations.pytorch import ToTensorV2

sys.path.append("src")
import utils.image_tools as T
from utils.training import create_augmentation

class BaseDataset(ABC, Dataset):
    def __init__(
        self,
        data_path: str | os.PathLike,
        target: str = "train",
        transform: A.Compose | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        # Initialize samples by reading them from the target path
        self.samples = self.init_samples(os.path.join(data_path, target))

        # Initialize custom transform
        self.transform = transform
        
        if seed is not None:
            # Sort the samples and shuffle them deterministically
            self.samples = sorted(self.samples, key=lambda x: x[0])
            random.seed(seed)
            random.shuffle(self.samples)
    
    @abstractmethod
    def init_samples(self, data_path: str) -> list[Any]:
        pass

    @abstractmethod
    def load_sample(self, *args) -> torch.Tensor | tuple[torch.Tensor]:
        pass

    def to_tensor(
        self,
        image: np.ndarray | Iterable[np.ndarray],
        normalize: bool | Iterable[bool] = True,
    ) -> torch.Tensor | list[torch.Tensor]:        
        if not isinstance(image, Iterable):
            # Generalize
            image = [image]

        if isinstance(normalize, bool):
            # If it is a single boolean, use it for every image
            normalize = [normalize for _ in range(len(image))]
        
        # A helper function to convert an image to a tensor with norm
        to_tensor_fn = lambda x: T.normalize(T.image_to_tensor(x[0])) \
                                 if x[1] else T.image_to_tensor(x[0])

        # Apply conversion method to all images (could be just one)
        tensors = list(map(to_tensor_fn, zip(image, normalize)))
        
        return tensors[0] if len(tensors) == 1 else tensors
    
    def load_transformable(
        self,
        image_path: str | os.PathLike | Iterable[str | os.PathLike] = [],
        mask_path: str | os.PathLike | Iterable[str | os.PathLike] = [],
    ) -> dict[str, np.ndarray]:
        if not isinstance(image_path, Iterable):
            # Generalize as a list
            image_path = [image_path]
        
        if not isinstance(mask_path, Iterable):
            # Generalize as a list
            mask_path = [mask_path]
        
        # Load a dictionary of samples in the albumentations dict form
        samples = {
            **{
                f"image{i if i > 0 else ''}": T.load_image(path) 
                for i, path in enumerate(image_path)
            },
            **{
                f"mask{i if i > 0 else ''}": T.load_image(path, True) 
                for i, path in enumerate(mask_path)
            }
        }

        return samples
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        # Get the stored sample(-s)
        sample = self.samples[index]
        sample = [sample] if not isinstance(sample, Iterable) else sample
        
        # Load the samples and labels
        loaded = self.load_sample(*sample)
        
        return loaded
    
    def __len__(self) -> int:
        return len(self.samples)

class BaseDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_class: BaseDataset,
                 data_path: str | os.PathLike,
                 augment_train: bool = True,
                 shuffle_val: bool = False,
                 **loader_kwargs):
        super().__init__()
        # Assign dataset attributes
        self.dataset_class = dataset_class
        self.data_path = data_path
        self.train_transform = None
        self.loader_kwargs = loader_kwargs
        self.shuffle_val = shuffle_val

        if augment_train:
            # Create a default augmentation composition
            self.train_transform = create_augmentation(additional_targets=2)

        # Set some default data loader arguments
        self.loader_kwargs.setdefault("batch_size", 10)
        self.loader_kwargs.setdefault("num_workers", 10)
        self.loader_kwargs.setdefault("pin_memory", True)

    def train_dataloader(self) -> DataLoader:
        # Create train dataset and return loader
        train_dataset = self.dataset_class(
            data_path=self.data_path,
            target="train",
            transform=self.train_transform
        )
        return DataLoader(train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        # Create val dataset and return loader
        val_dataset = self.dataset_class(
            data_path=self.data_path,
            target="val",
            seed=0
        )
        return DataLoader(val_dataset, shuffle=self.shuffle_val, **self.loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        # Create test dataset and return loader
        test_dataset = self.dataset_class(
            data_path=self.data_path,
            target="test",
            seed=0
        )
        return DataLoader(test_dataset, **self.loader_kwargs)