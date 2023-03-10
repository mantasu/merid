import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import MobileNetV2, InpaintGenerator

class LafinInpainter(nn.Module):
    def __init__(self,
                 det_weights: str | None = None,
                 gen_weights: str | None = None,
                 freeze: bool | None =None):
        super().__init__()

        # Initialize landmark detector & inpainter
        self.detector = MobileNetV2(points_num=68)
        self.generator = InpaintGenerator()

        if freeze is None:
            # Freeze is true if weights are provided, otherwise false
            freeze = det_weights is not None and gen_weights is not None

        if det_weights is not None:
            # Load the appropriate detector weights form the specified path
            self.detector.load_state_dict(torch.load(det_weights)["detector"])

        if gen_weights is not None:
            # Load the appropriate generator weights form the specified path
            self.generator.load_state_dict(torch.load(gen_weights)["generator"])
        
        if freeze:
            for param in self.parameters():
                # Disable gradient computation
                param.requires_grad = False
            
            # Eval mode
            self.eval()


    def interpolate(self, masks, factor):
        # Create the interpolation size and interppolate
        size = (masks.shape[2] // factor, masks.shape[3] // factor)
        return F.interpolate(masks, size=size, mode="bilinear", align_corners=True)
    
    def forward(self, images, masks):
        # Overlay the masks on the images
        images_masked = (1 - masks) * images + masks

        # Predict landmark coordinates, convert to appropriate type
        coords = (self.detector(images_masked) * min(images.shape[2:])).long()
        coords = coords.view(-1, 68, 2).clip(0, min(images.shape[2:]) - 1)
        
        # Generate landmarks from coordinates
        landmarks = torch.zeros_like(images[:, :1])
        landmarks[:, 0, coords[:, :, 1], coords[:, :, 0]] = 1

        # Create image inpainter/generator inputs
        inputs = torch.cat((images_masked, landmarks), dim=1)
        masks_quarter = self.interpolate(masks, 4)
        masks_half = self.interpolate(masks, 2)

        # Predict the inpainted images and create merged outputs
        images_out = self.generator(inputs, masks, masks_half, masks_quarter)
        outputs_merged = (images_out * masks) + (images * (1 - masks))

        return outputs_merged