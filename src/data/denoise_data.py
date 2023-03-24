import os
import torch
import albumentations as A
from .base_data import BaseDataset, BaseDataModule


class DenoiseDataset(BaseDataset):
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
        root_masks = os.path.join(data_path, "masks")
        root_glasses = os.path.join(data_path, "glasses")
        root_no_glasses = os.path.join(data_path, "no_glasses")
        samples = []
        
        for file in os.listdir(root_glasses):
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


class DenoiseDataModule(BaseDataModule):
    def __init__(self, **kwargs):
        super().__init__(DenoiseDataset, **kwargs)
