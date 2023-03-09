import pytorch_lightning as pl
from .ddnm.ddnm_inpainter import DDNMInpainter
from .nafnet.artefact_remover import NAFNetArtefactRemover

class MaskInpainter(pl.LightningModule):
    def __init__(self, diff, config={}):
        super().__init__()

        self.core_inpainter = config.get("core_inpainter", None)
        self.post_processer = config.get("post_processer", None)
        self.discriminator = config.get("discriminator", None)

        # self.is_ci_frozen = all(not p.requires_grad for p in self.core_inpainter.parameters())
    
    def forward(self, img, mask, feat=None):
        if self.core_inpainter is not None:
            feat = self.core_inpainter(img, mask)

        if self.post_processer is not None:
            feat = self.post_processer(img, mask, feat)
        
        return feat
