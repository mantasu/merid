import os
import sys
import torch
import random
import albumentations as A

from typing import Any, Iterable
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from albumentations.pytorch import ToTensorV2

class BaseDataset(ABC, Dataset):
    def __init__(self,
                 data_path: str | os.PathLike,
                 target: str ="train",
                 transform: A.Compose | None = None,
                 seed: int | None = None):
        super().__init__()

        # Initialize samples by reading them from the target path
        self.samples = self.init_samples(os.path.join(data_path, target))

        # Initialize custom transform
        self.transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ]) if transform is None else transform
        
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
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        # Get the stored sample(-s)
        sample = self.samples[index]
        sample = [sample] if not isinstance(sample, Iterable) else sample
        
        # Load the samples and labels
        loaded = self.load_sample(*sample)
        
        return loaded
    
    def __len__(self) -> int:
        return len(self.samples)
