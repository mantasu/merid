import torch
import pytorch_lightning as pl
from .ddnm.ddnm_inpainter import DDNMInpainter
from .nafnet.artefact_remover import NAFNetArtefactRemover

class MaskInpainter(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.core_inpainter = config["core_inpainter"]
        self.post_processer = config.get("post_processer", None)
        self.discriminator = config.get("discriminator", None)

        # self.is_ci_frozen = all(not p.requires_grad for p in self.core_inpainter.parameters())
    
    def forward(self, img, mask, feat=None):
        img_new = self.core_inpainter(img, mask)

        if self.post_processer is not None:
            feat = self.post_processer(img, mask, feat)
        
        return feat
    
    def training_step(self, batch, batch_index):
        (images_synthetic_glasses, images_synthetic_no_glasses,
         masks_synthetic, images_celeba, masks_celeba) = batch
        
        images_synthetic_inpainted = self.core_inpainter(images_synthetic_glasses, masks_synthetic)
        images_celeba_inpainted = self.core_inpainter(images_celeba, masks_celeba)

        images_synthetic_inpainted = self.post_processer(images_synthetic_inpainted, masks_synthetic)
        images_celeba_inpainted = self.post_processer(images_celeba_inpainted, masks_)


    def training_step_post_processer():
        pass

    def configure_optimizers(self):
        if self.post_processer is None or self.discriminator is None:
            # If post processer is not used
            return []
        
        # Get optimizers for discriminator and for generator
        optimizers = [
            self.optimizer_g(self.post_processer.parameters()),
            self.optimizer_d(self.discriminator.parameters())
        ]

        return optimizers
