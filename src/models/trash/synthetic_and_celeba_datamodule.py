import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from models.trash.synthetic_dataset import SyntheticDataModule
from glasses_and_not_dataset import GlassesAndNotDataModule

class SyntheticAndCelebaDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_path_synthetic: str | os.PathLike = "data/synthetic",
                 data_path_celeba: str | os.PathLike = "data/celeba",
                 augment_train: bool = True,
                 **loader_kwargs):
        self.synthetic_module = SyntheticDataModule(
            data_path_synthetic, augment_train, **loader_kwargs)
        self.celeba_datamodule = GlassesAndNotDataModule(
            data_path_celeba, augment_train, **loader_kwargs)
    
    def train_dataloader(self) -> list[DataLoader, DataLoader]:
        synthetic_dataloader = self.synthetic_module.train_dataloader()
        celeba_dataloader = self.celeba_datamodule.train_dataloader()

        return [synthetic_dataloader, celeba_dataloader]
    
    def val_dataloader(self) -> list[DataLoader, DataLoader]:
        synthetic_dataloader = self.synthetic_module.val_dataloader()
        celeba_dataloader = self.celeba_datamodule.val_dataloader()

        return [synthetic_dataloader, celeba_dataloader]
    
    def test_dataloader(self) -> list[DataLoader, DataLoader]:
        synthetic_dataloader = self.synthetic_module.test_dataloader()
        celeba_dataloader = self.celeba_datamodule.test_dataloader()

        return [synthetic_dataloader, celeba_dataloader]
