import os
import torch
import albumentations as A
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from .base_data import BaseDataset, BaseDataModule


class DenoiseSyntheticDataset(BaseDataset):
    def __init__(
        self,
        data_path: bool = "data/synthetic",
        target: str = "train",
        transform: A.Compose | None = None,
        frames_every_n: int = 5,
        seed: int = 0,
    ):
        self.frames_every_n = frames_every_n
        super().__init__(data_path, target, transform, seed)
    
    def init_samples(self, data_path: str) -> list[str]:
        # Initialize root paths and a list of samples
        root_masks = os.path.join(data_path, "masks")
        root_glasses = os.path.join(data_path, "glasses")
        root_no_glasses = os.path.join(data_path, "no_glasses")
        samples, i = [], 0
        
        for file in os.listdir(root_glasses):
            if file.endswith("-all.jpg") and i < self.frames_every_n:
                # Skip
                i += 1
                continue
            else:
                # Reset
                i = 0

            if file.endswith("-all.jpg"):
                # If normal eyeglasses, use mask frame
                file_mask = file.replace("-all", "-mask-frame")
                file_no_glasses = file.replace("-all", "-face")
                file_inpainting = file.replace("-all", "-inpainted-frame")
            else:
                # If sunglasses use full mask over eye region
                file_mask = file.replace("-sunglasses", "-mask-full")
                file_no_glasses = file.replace("-sunglasses", "-face")
                file_inpainting = file.replace("-sunglasses", "-inpainted-full")

            # Join some terms to create full paths to images
            path_mask = os.path.join(root_masks, file_mask)
            path_glasses = os.path.join(root_glasses, file)
            path_no_glasses = os.path.join(root_no_glasses, file_no_glasses)
            path_inpainting = os.path.join(root_no_glasses, file_inpainting)

            # Append the image, inpainting, mask and ground-truth
            samples.append([path_glasses, path_inpainting,
                            path_no_glasses, path_mask])

        return samples
    
    def load_sample(self,
        path_glasses: str,
        path_inpainting: str,
        path_no_glasses: str,
        path_mask: str,
    ) -> tuple[torch.Tensor]:
        # Load samples to a transformable (albumentations) dictionary
        image_paths = (path_glasses, path_inpainting, path_no_glasses)
        sample = self.load_transformable(image_paths, path_mask)

        if self.transform is not None:
            # Apply any transforms if needed
            sample = self.transform(**sample)
        
        # Convert everything to tensor, only normalize image and inpaint
        samples = self.to_tensor(sample.values(), [True, True, False, False])

        return tuple(samples)


class DenoiseCelebADataset(BaseDataset):
    def __init__(
        self,
        data_path: bool = "data/celeba",
        target: str = "train",
        transform: A.Compose | None = None,
        seed: int = 0,
    ):
        super().__init__(data_path, target, transform, seed)
    
    def init_samples(self, data_path: str) -> list[str]:
        # Initialize root paths and a list of samples
        root_masks = os.path.join(data_path, "masks")
        root_glasses = os.path.join(data_path, "glasses")
        root_no_glasses = os.path.join(data_path, "no_glasses")
        samples, i = [], 0
        
        for file in os.listdir(root_glasses):
            # Create corresponding file names
            file_mask = file[:-4] + "-mask.jpg"
            file_inpainting = file[:-4] + "-inpainted.jpg"

            # Join some terms to create full paths to images
            path_mask = os.path.join(root_masks, file_mask)
            path_glasses = os.path.join(root_glasses, file)
            path_inapint = os.path.join(root_no_glasses, file_inpainting)

            # Append the image, inpainting, mask and ground-truth
            samples.append([path_glasses, path_inapint, path_mask])
        
        for file in os.listdir(root_no_glasses):
            if file.endswith("-inpainted.jpg"):
                # Skip inpainted images
                continue

            # Add a random image without glasses (since no ground truth)
            samples[i].insert(2, os.path.join(root_no_glasses, file))
            i += 1

            if i == len(samples):
                # Break if no more
                break

        return samples
    
    def load_sample(self,
        path_glasses: str,
        path_inpainting: str,
        path_no_glasses: str,
        path_mask: str,
    ) -> tuple[torch.Tensor]:
        # Load samples to a transformable (albumentations) dictionary
        image_paths = (path_glasses, path_inpainting, path_no_glasses)
        sample = self.load_transformable(image_paths, path_mask)

        if self.transform is not None:
            # Apply any transforms if needed
            sample = self.transform(**sample)
        
        # Convert everything to tensor, only normalize image and inpaint
        samples = self.to_tensor(sample.values(), [True, True, False, False])

        return tuple(samples)


class DenoiseSyntheticDataModule(BaseDataModule):
    def __init__(self, **kwargs):
        super().__init__(DenoiseSyntheticDataset, shuffle_val=True, **kwargs)


class DenoiseCelebADataModule(BaseDataModule):
    def __init__(self, **kwargs):
        super().__init__(DenoiseCelebADataset, shuffle_val=True, **kwargs)


class DenoiseDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

        # Initialize the synthetic and celeba sub-datamodules
        self.synthetic_datamodule = DenoiseSyntheticDataModule(**kwargs)
        self.celeba_datamodule = DenoiseCelebADataModule(**kwargs)
    
    def train_dataloader(self) -> dict[str, DataLoader]:
        # Get both train dataloaders for synthetic and celeba datasets
        synthetic_dataloader = self.synthetic_datamodule.train_dataloader()
        celeba_dataloader = self.celeba_datamodule.train_dataloader()

        return {"synthetic": synthetic_dataloader, "celeba": celeba_dataloader}

    def val_dataloader(self) -> list[DataLoader]:
        # Get both val dataloaders for synthetic and celeba datasets
        synthetic_dataloader = self.synthetic_datamodule.val_dataloader()
        celeba_dataloader = self.celeba_datamodule.val_dataloader()
        
        return [synthetic_dataloader, celeba_dataloader]

    def test_dataloader(self) -> list[DataLoader]:
        # Get both test dataloaders for synthetic and celeba datasets
        synthetic_dataloader = self.synthetic_datamodule.test_dataloader()
        celeba_dataloader = self.celeba_datamodule.test_dataloader()
        
        return [synthetic_dataloader, celeba_dataloader]
