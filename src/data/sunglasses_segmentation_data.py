import os
import torch
import albumentations as A
from .base_data import BaseDataset, BaseDataModule


class SunglassesSegmentationDataset(BaseDataset):
    def __init__(
        self,
        data_path: str | os.PathLike = "data/celeba-mask-hq",
        **kwargs
    ):
        super().__init__(data_path, **kwargs)

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
        sample = self.load_transformable(image_path, mask_path)

        if self.transform is not None:
            # Apply any transforms if needed
            sample = self.transform(**sample)

        # Convert to tensor, normalize only the image, binarize masks
        samples = self.to_tensor(sample.values(), normalize=[True, False])
        samples[1] = samples[1].round().float()

        return tuple(samples)


class SunglassesSegmentationDataModule(BaseDataModule):
    def __init__(self, **kwargs):
        super().__init__(SunglassesSegmentationDataset, **kwargs)
        
        # Reset the default parameters for data loader
        self.loader_kwargs.setdefault("batch_size", 12)
        self.loader_kwargs.setdefault("num_workers", 6)
