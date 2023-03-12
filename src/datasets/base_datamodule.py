import os
import sys
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from base_dataset import BaseDataset

sys.path.append("src")
from utils.augment import create_default_augmentation

class BaseDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_class: BaseDataset,
                 data_path: str | os.PathLike,
                 augment_train: bool = True,
                 **loader_kwargs):
        super().__init__()
        # Assign dataset attributes
        self.dataset_class = dataset_class
        self.data_path = data_path
        self.train_transform = None
        self.loader_kwargs = loader_kwargs

        if augment_train:
            # Create a default augmentation composition
            self.train_transform = create_default_augmentation()

        # Set some default data loader arguments
        self.loader_kwargs.setdefault("batch_size", 10)
        self.loader_kwargs.setdefault("num_workers", 24)
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
        return DataLoader(val_dataset, **self.loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        # Create test dataset and return loader
        test_dataset = self.dataset_class(
            data_path=self.data_path,
            target="test",
            seed=0
        )
        return DataLoader(test_dataset, **self.loader_kwargs)
