import os
import torch
import random
import albumentations as A
import torchvision.transforms as T

from typing import Sequence
from torch.utils.data import Dataset
from glasses_labeled_dataset import GlassesLabeledDataset
from glasses_and_no_glasses_mix_dataset import GlassesAndNoGlassesMixDataset

class LabeledAndNonLabeledMixDataset(Dataset):
    def __init__(self,
                 path_x1: str | os.PathLike,
                 path_x2: str | os.PathLike,
                 path_y1: str | os.PathLike | None = None,
                 path_y2: str | os.PathLike | None = None,
                 return_labels: bool = True,
                 replacable_ending: str = "-all",
                 label_identifiers: Sequence[tuple[str, bool]] = [],
                 transform: A.Compose | T.Compose | None = None,
                 seed: int | None = None,
                 load_in_memory: bool = False
                ):
        super().__init__()

        # Create the labeled and non-labeled datasets
        self.labeled_dataset = GlassesLabeledDataset(
            path_x1, path_y1, replacable_ending, label_identifiers, transform, load_in_memory
        )
        self.non_labeled_dataset = GlassesAndNoGlassesMixDataset(
            path_x2, path_y2, return_labels, transform, seed, load_in_memory
        )
        
        # Generate default non-primary return values
        self.default_labeled = [
            torch.empty_like(tensor) for tensor in
            self.verify_sequence(self.non_labeled_dataset[0])[1:]
        ]
        self.default_non_labeled = [
            torch.empty_like(tensor) for tensor in 
            self.verify_sequence(self.labeled_dataset[0])[1:]
        ]
        
        # Create a joint indices list
        self.entry_indices = [
            *range(-len(self.labeled_dataset), 0),
            *range(len(self.non_labeled_dataset))
        ]

        if seed is not None:
            # Seed if needed
            random.seed(seed)
        
        # Shuffle entry indices in-place
        random.shuffle(self.entry_indices)
    
    def verify_sequence(self, sequence: torch.Tensor |
                        Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if torch.is_tensor(sequence):
            # Convert to sequence
            sequence = [sequence]
        
        return sequence
    
    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor]:
        # Get index from a shuffled indices list
        index = self.entry_indices(index)

        if index < 0:
            # If less than 0, then index belongs to the labeled dataset
            sample = self.verify_sequence(self.labeled_dataset[abs(index + 1)])
            sample = (*sample, *self.default_non_labeled)
        else:
            # If more than 0, then index belongs to the non-labeled dataset
            sample = self.verify_sequence(self.non_labeled_dataset[index])
            sample = (*self.default_labeled, *sample)
        
        return sample[0] if len(sample) == 1 else sample

    def __len__(self) -> int:
        return len(self.labeled_dataset) + len(self.non_labeled_dataset)
