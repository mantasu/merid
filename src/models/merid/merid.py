import sys
import torch
import torch.nn as nn

from PIL import Image
from typing import Any
from skimage.morphology import disk
from kornia.morphology import dilation, closing

from ..pesr.domain_adaption import DomainAdapter
from ..pesr.segmentation import ResnetGeneratorMask
from ..ddnm.ddnm_inpainter import DDNMInpainter
from ..lafin.lafin_inpainter import LafinInpainter
from ..nafnet.nafnet_denoiser import NAFNetDenoiser
from .recolorizer import Recolorizer
from .sunglasses_classifier import SunglassesClssifier
from .sunglasses_segmenter import GlassesSegmenter

sys.path.append("src")

from utils.image_tools import normalize, image_to_tensor, tensor_to_image, load_image, unnormalize

class MaskGenerator(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()

        # Initialize the domain adapter, and the two mask segmenters
        self.domain_adapter: DomainAdapter = config["domain_adapter"]
        self.glasses_masker: ResnetGeneratorMask = config["glasses_masker"]
        self.shadows_masker: ResnetGeneratorMask = config["shadows_masker"]
    
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor]:
        # Get the feature vectors for glasses and shadows
        f_glasses, f_shadows = self.domain_adapter(image)

        # Get the glasses output and convert a mask
        out_glasses = self.glasses_masker(f_glasses)
        mask_glasses = out_glasses.argmax(1).unsqueeze(1).float()

        # Update shadow feature and get shadows output for mask
        f_shadows = torch.cat([f_shadows, mask_glasses], dim=1)
        out_shadows = self.shadows_masker(f_shadows)
        mask_shadows = out_shadows.argmax(1).unsqueeze(1).float()

        return mask_glasses, mask_shadows


class MaskRetoucher(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()

        # Initialize the sunglasses classifier and the full segmenter
        self.classifier: SunglassesClssifier = config["sunglasses_predictor"]
        self.segmenter: GlassesSegmenter = config["sunglasses_segmenter"]

        # Assign morphological disk kernels for dilation and closing
        kernel_closing = torch.from_numpy(disk(3)).float()
        kernel_dilation = torch.from_numpy(disk(1)).float()
        self.register_buffer("kernel_closing", kernel_closing)
        self.register_buffer("kernel_dilation", kernel_dilation)

    def forward(
        self,
        image: torch.Tensor,
        mask_glasses: torch.Tensor,
        mask_shadows: torch.Tensor
    ) -> torch.Tensor:
        
        # Encance sunglasses masks, apply morphology and join both masks
        mask_glasses = self.enhance_mask_if_sunglasses(image, mask_glasses)
        masks = self.apply_morphology(mask_glasses, mask_shadows)
        mask = torch.logical_or(*masks).float()
        
        return mask
    
    def enhance_mask_if_sunglasses(
        self,
        image: torch.Tensor,
        mask_glasses: torch.Tensor
    ) -> torch.Tensor:
        # Predict if sunglasses & convert to bool as sunglasses indices
        is_sungl = self.classifier(image).flatten().sigmoid().round().bool()

        if is_sungl.sum() > 0:
            # Get sunglasses sub-batch and predict full sunglasses mask
            image_sub, mask_sub = image[is_sungl], mask_glasses[is_sungl]
            mask_sunglasses = self.segmenter(image_sub).sigmoid().round()

            # Update sub-batch of sunglasses to include full masks
            mask_sub = torch.logical_or(mask_sunglasses, mask_sub)
            mask_glasses[is_sungl] = mask_sub.float()
        
        return mask_glasses
    
    def apply_morphology(self, *masks) -> torch.Tensor:
        # Init return vals
        outputs = tuple()

        for mask in masks:
            # Perform small dilation and bigger closing
            mask = dilation(mask, self.kernel_dilation)
            mask = closing(mask, self.kernel_closing)
            outputs = (*outputs, mask)
        
        return outputs


class MaskInpainter(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        
        # Initialize one of the inpainters, NAFNet denoiser, recolorizer
        self.inpainter: LafinInpainter | DDNMInpainter = config["inpainter"]
        self.denoiser: NAFNetDenoiser = config["denoiser"]
        self.recolorizer: Recolorizer = config["recolorizer"]

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Inpaint, denoise grayscale, colorize
        inpainted = self.inpainter(unnormalize(image), mask)
        grayscale = self.denoiser(image, normalize(inpainted), mask)
        colorized = self.recolorizer(grayscale, image)
        
        return colorized


class Merid(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()

        # Initialize the three main building blocks from the paper
        self.mask_generator = MaskGenerator(config["mask_generator"])
        self.mask_retoucher = MaskRetoucher(config["mask_retoucher"])
        self.mask_inpainter = MaskInpainter(config["mask_inpainter"])
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Apply the three main building blocks in a sequence
        mask_glasses, mask_shadows = self.mask_generator(image)
        mask = self.mask_retoucher(image, mask_glasses, mask_shadows)
        image = self.mask_inpainter(image, mask)

        return image
    
    def predict(self, img: str | Image.Image | list[str | Image.Image]
                ) -> Image.Image:
        if isinstance(img, (str, Image.Image)):
            # Generalize
            img = [img]

        # Device will be needed to perfom operations on
        device = next(iter(self.parameters())).device
        
        # Convert every image path to an actual image, then to a tensor
        img = [load_image(i) if isinstance(i, str) else i for i in img]
        img = torch.stack([image_to_tensor(i, device=device) for i in img])

        # Predict and convert
        img = self(normalize(img))
        img = [tensor_to_image(i, as_pil=True) for i in img]

        return img[0] if len(img) == 1 else img
