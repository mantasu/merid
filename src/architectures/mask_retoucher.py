import torch
import torch.nn as nn
from skimage.morphology import disk
from kornia.morphology import dilation, closing

class MaskRetoucher(nn.Module):
    def __init__(self, config: dict[str, nn.Module, bool]):
        super().__init__()

        # Setup sunglasses classifier, segmenter and morphology kernels
        self.sunglasses_predictor = config.get("sunglasses_predictor", None)
        self.sunglasses_segmenter = config.get("sunglasses_segmenter", None)
        self.kernel_closing = torch.from_numpy(disk(3)).float()
        self.kernel_dilation = torch.from_numpy(disk(1)).float()

        if config.get("freeze_sunglasses_predictor", False):
            for param in self.sunglasses_predictor.params():
                # Disable gradients to freeze
                param.requires_grad = False
            
            # Set to evaluation mode if frozen
            self.sunglasses_predictor.eval()
        
        if config.get("freeze_sunglasses_segmenter", False):
            for param in self.sunglasses_segmenter.params():
                # Disable gradients to freeze
                param.requires_grad = False
            
            # Set to evaluation mode if frozen
            self.sunglasses_segmenter.eval()

    def forward(self, x: torch.Tensor, mask_glasses: torch.Tensor,
                mask_shadows: torch.Tensor) -> torch.Tensor:
        # Predict sunglasses mask if needed, enhance and join both masks
        mask_glasses = self.enhance_mask_if_sunglasses(x, mask_glasses)
        masks = self.apply_morphology(mask_glasses, mask_shadows)
        mask = torch.logical_or(*masks)

        return mask
    
    def enhance_mask_if_sunglasses(self, x, mask_glasses: torch.Tensor) -> torch.Tensor:
        if self.sunglasses_predictor is None or self.sunglasses_segmenter is None:
            # No sunglasses exist
            return mask_glasses
        
        # Predict if sunglasses and convert to bool as indices
        is_sunglasses = self.sunglasses_predictor(x).flatten()
        is_sunglasses = is_sunglasses.sigmoid().round().bool()

        if is_sunglasses.sum() > 0:
            # Get the sub-batch of samples with sunglasses and join masks
            mask_sunglasses = self.sunglasses_segmenter(x[is_sunglasses])
            mask_glasses = mask_glasses | mask_sunglasses

            # mask_sunglasses = self.sunglasses_segmenter(x[is_sunglasses])
            # mask_glasses = (mask_glasses.bool() | mask_sunglasses.bool()).float()
        
        return mask_glasses
    
    def apply_morphology(self, *masks):
        # Init return vals
        outputs = tuple()

        for mask in masks:
            # Perform small dilation and bigger closing
            mask = dilation(mask, self.kernel_dilation)
            mask = closing(mask, self.kernel_closing)
            outputs = (*outputs, mask)
        
        return outputs
