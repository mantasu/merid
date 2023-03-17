import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from torch.utils.data import Subset

import sys
sys.path.append("src")

from utils.io import load_image
from utils.augment import create_default_augmentation, unnormalize
from PIL import Image

class ColorizeDataset(Dataset):
    def __init__(self, syn_path="data/synthetic", celeba_path="data/celeba", target="train", transform = None):
        super().__init__()

        samples = []

        self.transform = A.Compose([A.Normalize(), ToTensorV2()], additional_targets={"no_glasses": "image"}) if transform is None else transform

        root_syn = os.path.join(syn_path, target)
        # root_cel = os.path.join(celeba_path, target)
        

        for file in os.listdir(os.path.join(root_syn, "glasses")):
            
            img_glas = os.path.join(root_syn, "glasses", file)
            img_sun_glass = os.path.join(root_syn, "sunglasses", file.replace("-all", "-sunglasses"))
            img_no_glass = os.path.join(root_syn, "no_glasses", file.replace("-all", "-face"))

            samples.append([img_glas, img_no_glass])
            samples.append([img_sun_glass, img_no_glass])
        
        random.seed(0)
        random.shuffle(samples)

        self.samples = samples
    
    def load_sample(self, sample):
        samples = self.transform(image=load_image(sample[0]), no_glasses=load_image(sample[1]))
        # samples["image"] = unnormalize(samples["image"])
        samples["no_glasses"] = unnormalize(samples["no_glasses"])

        return tuple(samples.values())

    def __getitem__(self, index):
        return self.load_sample(self.samples[index])
    
    def __len__(self):
        return len(self.samples)
    
class ColorizeDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_transform = None
        self.loader_kwargs = {}

        
        # Create a default augmentation composition
        self.train_transform = create_default_augmentation()

        # Set some default data loader arguments
        self.loader_kwargs.setdefault("batch_size", 10)
        self.loader_kwargs.setdefault("num_workers", 10)
        self.loader_kwargs.setdefault("pin_memory", True)

    def train_dataloader(self) -> DataLoader:
        # Create train dataset and return loader
        train_dataset = ColorizeDataset(
            target="train",
            transform=self.train_transform
        )
        return DataLoader(train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        # Create val dataset and return loader
        val_dataset = ColorizeDataset(
            target="val",
        )

        return DataLoader(val_dataset, shuffle=True, **self.loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        # Create test dataset and return loader
        test_dataset = ColorizeDataset(
            target="test",
        )

        return DataLoader(test_dataset, **self.loader_kwargs)