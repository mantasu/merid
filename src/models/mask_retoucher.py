import torch
import torch.nn as nn
import pytorch_lightning as pl
from skimage.morphology import disk
from kornia.morphology import dilation, closing

class MaskRetoucher(pl.LightningModule):
    def __init__(self, config: dict[str, nn.Module, bool]):
        super().__init__()

        # Setup sunglasses classifier, segmenter and morphology kernels
        self.sunglasses_predictor = config.get("sunglasses_predictor", None)
        self.sunglasses_segmenter = config.get("sunglasses_segmenter", None)
        self.register_buffer("kernel_closing", torch.from_numpy(disk(3)).float())
        self.register_buffer("kernel_dilation", torch.from_numpy(disk(1)).float())
        self.segment_when = "only_if_sunglasses"

    def forward(self, x: torch.Tensor, mask_glasses: torch.Tensor,
                mask_shadows: torch.Tensor) -> torch.Tensor:
        
        is_sunglasses = None

        match self.segment_when:
            case "only_if_sunglasses":
                mask_glasses, is_sunglasses = self.enhance_mask_if_sunglasses(x, mask_glasses)
            case "always":
                mask_glasses = torch.logical_or(self.sunglasses_segmenter(x).sigmoid().round(), mask_glasses).float()
            case "never":
                mask_glasses = mask_glasses
        
        masks = self.apply_morphology(mask_glasses, mask_shadows)
        mask = torch.logical_or(*masks).float()
        
        return mask, is_sunglasses
    
    def enhance_mask_if_sunglasses(self, x: torch.Tensor, mask_glasses: torch.Tensor) -> torch.Tensor:
        if self.sunglasses_predictor is None or self.sunglasses_segmenter is None:
            # No sunglasses exist
            return mask_glasses, None
        
        # Predict if sunglasses and convert to bool as indices
        is_sunglasses = self.sunglasses_predictor(x).flatten()
        is_sunglasses = is_sunglasses.sigmoid().round().bool()

        if is_sunglasses.sum() > 0:
            # Get the sub-batch of samples with sunglasses and update glasses
            img_sub, msk_sub = x[is_sunglasses], mask_glasses[is_sunglasses]
            mask_sunglasses = self.sunglasses_segmenter(img_sub).sigmoid()
            msk_sub = torch.logical_or(mask_sunglasses.round(), msk_sub)
            mask_glasses[is_sunglasses] = msk_sub.float()
        
        return mask_glasses, is_sunglasses
    
    def apply_morphology(self, *masks):
        # Init return vals
        outputs = tuple()

        for mask in masks:
            # Perform small dilation and bigger closing
            mask = dilation(mask, self.kernel_dilation)
            mask = closing(mask, self.kernel_closing)
            outputs = (*outputs, mask)
        
        return outputs
