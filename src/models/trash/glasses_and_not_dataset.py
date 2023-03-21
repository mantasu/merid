import os
import torch
import albumentations as A

from base_data import BaseDataset
from base_datamodule import BaseDataModule
from utils.io import load_image

class GlassesAndNotDataset(BaseDataset):
    def __init__(self,
                 data_path: str | os.PathLike = "data/celeba",
                 target: str ="train",
                 transform: A.Compose | None = None,
                 seed: int | None = None
                ):
        super().__init__(data_path, target, transform, seed)
    
    def init_samples(self, data_path: str) -> list[str, int]:
        samples = []

        for file in os.listdir(root := os.path.join(data_path, "no_glasses")):
            samples.append((os.path.join(root, file), 0))

        for file in os.listdir(root := os.path.join(data_path, "glasses")):
            samples.append((os.path.join(root, file), 1))

        return samples

    def load_sample(self, sample_path: str, label: int) -> tuple[torch.Tensor]:
        image = load_image(sample_path)
        image = self.transform(image=image)["image"]
        label = torch.tensor(label).float()

        return image, label


class GlassesAndNotDataModule(BaseDataModule):
    def __init__(self,
                 data_path: str | os.PathLike = "data/celeba",
                 augment_train: bool = True,
                 **loader_kwargs):
        super().__init__(GlassesAndNotDataset, data_path, augment_train, **loader_kwargs)
