import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split

class GeneralModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, loader_kwargs={}):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loader_kwargs = loader_kwargs

    def train_dataloader(self):
        # Create a DataLoader for the training set
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self):
        # Create a DataLoader for the validation set
        return DataLoader(self.val_dataset, shuffle=False, **self.loader_kwargs)