import os
import torch
import albumentations as A
from .base_data import BaseDataset, BaseDataModule


class RecolorizeDataset(BaseDataset):
    def __init__(
        self,
        data_path: bool = "data/synthetic",
        target: str = "train",
        transform: A.Compose | None = None,
        seed: int = 0,
    ):
        super().__init__(data_path, target, transform, seed)
    
    def init_samples(self, data_path: str) -> list[str]:
        # Initialize root paths and a list of samples
        root_glasses = os.path.join(data_path, "glasses")
        root_no_glasses = os.path.join(data_path, "no_glasses")
        samples = []

        for file in os.listdir(root_glasses):
            # Replace name with glasses identifiers no-glasses path
            file_no_glasses = file.replace("-sunglasses", "-face")
            file_no_glasses = file_no_glasses.replace("-all", "-face")

            # Join the roots and filenames to form paths
            path_glasses = os.path.join(root_glasses, file)
            path_no_glasses = os.path.join(root_no_glasses, file_no_glasses)

            # Append the sample and the ground-truth paths
            samples.append((path_glasses, path_no_glasses))
        
        return samples
    
    def load_sample(self, glasses_path: str, no_glasses_path: str
                    ) -> tuple[torch.Tensor]:
        # Load samples to a transformable (albumentations) dictionary
        sample = self.load_transformable((glasses_path, no_glasses_path))

        if self.transform is not None:
            # Apply any transforms if needed
            sample = self.transform(**sample)

        # Convert to tensor, normalize only glasses image, add grayscale
        samples = self.to_tensor(sample.values(), normalize=[True, False])
        samples.insert(0, samples[1].mean(dim=0, keepdim=True))

        return tuple(samples)


class RecolorizeDataModule(BaseDataModule):
    def __init__(self, **kwargs):
        super().__init__(RecolorizeDataset, shuffle_val=True, **kwargs)
