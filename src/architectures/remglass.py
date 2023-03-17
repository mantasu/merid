import sys
import torch
import pytorch_lightning as pl

from typing import Any
from .mask_generator import MaskGenerator
from .mask_retoucher import MaskRetoucher
from .mask_inpainter import MaskInpainter

from torchvision.transforms.functional import normalize

sys.path.append("src")
from utils.augment import unnormalize

class RemGlass(pl.LightningModule):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        
        self.mask_generator = MaskGenerator(config["mask_generator"])
        self.mask_retoucher = MaskRetoucher(config["mask_retoucher"])
        self.mask_inpainter = None #  MaskInpainter(config["mask_inpainter"])

        # if config.get("freeze_mask_generator", False):
        #     self.mask_generator.freeze()
        #     self.mask_generator.eval()
        
        # if config.get("freeze_mask_retoucher", False):
        #     self.mask_retoucher.freeze()
        #     self.mask_retoucher.eval()
        
        # if config.get("freeze_mask_inpainter", False):
        #     self.mask_inpainter.freeze()
        #     self.mask_inpainter.eval()
        
        self.train_target = "inpainter" # Others not supported
    
    def forward_before_retoucher(self, x):
        # Renormalize to same scale as in paper, predict masks for glasses
        generator_input = normalize(unnormalize(x), [.5, .5, .5], [.5, .5, .5])
        out_glasses, out_shadows = self.mask_generator(generator_input)

        mask_glasses = out_glasses.argmax(1).unsqueeze(1).float()
        mask_shadows = out_shadows.argmax(1).unsqueeze(1).float()

        return mask_glasses, mask_shadows, out_glasses, out_shadows
    
    def forward_before_inpainter(self, x):
        # Get the 2 maps of the forward loop step before retoucher
        generator_out = self.forward_before_retoucher(x)
        mask_glasses, mask_shadows, out_glasses, out_shadows = generator_out
        mask, is_sunglasses = self.mask_retoucher(x, mask_glasses, mask_shadows)

        return mask, out_glasses, out_shadows, is_sunglasses
    
    def forward(self, x):
        out_retoucher = self.forward_before_inpainter(x)
        x_inpainted = self.mask_inpainter(x, *out_retoucher)

        return x_inpainted, out_retoucher[0]
    
    def process_inpainter_batch(self, batch):
        # batch:
        # 0) images_synthetic_glasses
        # 1) images_synthetic_no_glasses
        # 4) images_celeba
        # 5) y

        with torch.no_grad():
            masks_synthetic = self.forward_before_inpainter(batch[0])
            masks_celeba = torch.zeros_like(batch[3])

            if (idx := batch[4].bool().flatten()).sum() > 0:
                out = self.forward_before_inpainter(batch[4][idx])
                masks_celeba[idx], is_sunglasses = out[0], out[1]
            
        batch = (*batch[:2], *batch[4:], masks_synthetic, masks_celeba, is_sunglasses)
        
        return batch
    
    def training_step(self, batch):
        if self.train_target == "inpainter":
            batch = self.process_inpainter_batch(batch)
            loss = self.mask_inpainter.training_step(batch)

        return loss
    
    def validation_step(self, batch):
        return self.mask_inpainter.validation_step(batch)
    
    def validation_epoch_end(self, outputs):
        self.mask_inpainter.validation_epoch_end(outputs)

    def test_set(self, batch):
        return self.mask_inpainter.test_step(batch)
    
    def test_epoch_end(self, outputs):
        self.mask_inpainter.test_epoch_end(outputs)
    
    def configure_optimizers(self):
        return self.mask_inpainter.configure_optimizers()
