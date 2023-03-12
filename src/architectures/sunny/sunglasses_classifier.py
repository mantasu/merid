import os
import torch
import torch.nn as nn
from skimage.morphology import disk
from kornia.morphology import dilation, closing, erosion
from torchvision.models import shufflenet_v2_x0_5


class SunglassesClssifier(nn.Module):
    def __init__(self, weights_path: str | os.PathLike | None = None, freeze: bool | None = None):
        super().__init__()
        self.features = shufflenet_v2_x0_5()
        self.classifier = nn.Sequential(nn.Linear(1000, 1), nn.Flatten(0))

        if freeze is None:
            # Automatically determine whether to freeze
            freeze = weights_path is not None and weights_path != ""

        if weights_path is not None:
            weights = torch.load(weights_path)
            del weights["loss_fn.pos_weight"]
            self.load_state_dict(weights)

        if freeze:
            for param in self.parameters():
                # Freeze all the parameters
                param.requires_grad = False

            # Eval mode
            self.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

class MaskPostprocesser(nn.Module):
    def __init__(self, sunglasses_weights: str | os.PathLike = None):
        super().__init__()
        self.sunglasses_classifier = SunglassesClssifier(sunglasses_weights)

        self.non_sunglasses_structuring_element = torch.from_numpy(disk(1)).float()
        self.non_sunglasses_kernel = torch.ones_like(self.non_sunglasses_structuring_element)

        self.sunglasses_structuring_element = torch.from_numpy(disk(5)).float()
        self.sunglasses_kernel = torch.ones_like(self.sunglasses_structuring_element)

        self.closing_structuring_element = torch.from_numpy(disk(20)).float()
        self.closing_kernel = torch.ones_like(self.closing_structuring_element)
    
    def forward(self, image, mask1, mask2=None):
        is_sunglasses = self.sunglasses_classifier(image).sigmoid().round().bool()
        mask_enhanced = self.enhance_masks(mask1, is_sunglasses)

        if mask2 is not None:
            mask2 = dilation(mask2, self.non_sunglasses_kernel)
            masks_enhanced = mask_enhanced.logical_or(mask2)
        
        return masks_enhanced.long()
    
    def enhance_masks(self, masks, is_sunglasses):
        masks_enhanced = torch.empty_like(masks).float()

        masks_no_sunglasses = masks[~is_sunglasses]
        masks_sunglassses = masks[is_sunglasses]

        if len(masks_no_sunglasses) > 0:
            masks_no_sunglasses = dilation(masks_no_sunglasses, self.non_sunglasses_kernel)
            masks_enhanced[~is_sunglasses] = masks_no_sunglasses

        if len(masks_sunglassses) > 0:
            masks_sunglassses = dilation(masks_sunglassses, self.sunglasses_kernel)
            masks_sunglassses = closing(masks_sunglassses, self.closing_kernel)
            masks_enhanced[is_sunglasses] = masks_sunglassses
        
        return masks_enhanced
    
