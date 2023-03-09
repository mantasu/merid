import os
import cv2
import torch
import random
import albumentations as A
import torchvision.transforms as T

from torch.utils.data import Dataset

class GlassesAndNoGlassesMixDataset(Dataset):
    def __init__(self,
                 path_x: str | os.PathLike,
                 path_y: str | os.PathLike | None = None,
                 return_labels: bool = True,
                 transform: A.Compose | T.Compose | None = None,
                 seed: int | None = None,
                 load_in_memory: bool = False
                ):
        super().__init__()

        # Specify bool vars for getitem
        self.return_labels = return_labels
        self.load_in_memory = load_in_memory

        # Specify the transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((.5, .5, .5), (.5, .5, .5))
        ]) if transform is None else transform

        # List the files in the provided directories and make full paths
        paths_x = [os.path.join(path_x, x) for x in os.listdir(path_x)]
        paths_y = [os.path.join(path_y, y) for y in os.listdir(path_y)] \
                  if path_y is not None else []
        
        if return_labels:
            # Attach True/False labels for corresponding paths
            paths_x = list(map(lambda x: (x, True), paths_x))
            paths_y = list(map(lambda y: (y, False), paths_y))
        
        # Create a sorted list of all the paths
        self.samples = sorted(paths_x.extend(paths_y))
        
        if seed is not None:
            # Seed if needed
            random.seed(seed)
        
        # Shuffle samples in-place
        random.shuffle(self.samples)

        if load_in_memory:
            # Loaded all images in memory (for faster training)
            self.samples = list(map(self.path_to_tensor, self.samples))
    
    def __getitem__(self, index: int) -> torch.Tensor |\
                    tuple[torch.Tensor, torch.Tensor]:
        # Get the specified sample
        sample = self.samples[index]

        if not self.load_in_memory:
            # If not already loaded in memory
            sample = self.path_to_tensor(sample)
        
        return sample

    def __len__(self) -> int:
        return len(self.samples)

    def path_to_tensor(self, path: str | os.PathLike | \
                       tuple[str | os.PathLike, bool]) -> \
                       torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Read the image, convert to RGB and apply transforms
        img = cv2.imread(path[0] if self.return_labels else path)
        tensor = self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if isinstance(self.transform, A.Compose):
            # A.Compose uses keys
            tensor = tensor["image"]
        
        if self.return_labels:
            return tensor, torch.tensor(path[1], dtype=torch.long)
        else:
            return tensor