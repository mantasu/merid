import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms as T

from PIL import Image

from torchvision.models import (shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights, 
                                mobilenet_v3_small, MobileNet_V3_Small_Weights,
                                efficientnet_b0, EfficientNet_B0_Weights)

class SunglassesClssifier(pl.LightningModule):
    def __init__(self, num_epochs=-1, base_model: str = "shufflenet",
                 is_base_pretrained: bool = False):
        super().__init__()
        self.num_epochs = num_epochs
        self.base_model = self.load_base_model(base_model, is_base_pretrained)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]))

        # weights_path = True
        
        # if weights_path is not None:
        #     weights = torch.load(weights_path)
        #     del weights["loss_fn.pos_weight"]
        #     self.load_state_dict(weights)
    
    def load_base_model(self, model: str, is_pretrained: bool):
        match model:
            case "shufflenet":
                w = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
                m = shufflenet_v2_x0_5(weights=w if is_pretrained else None)
                m.fc = nn.Linear(m.fc.in_features, 1)
            case "mobilenet":
                w = MobileNet_V3_Small_Weights.IMAGENET1K_V1
                m = mobilenet_v3_small(weights=w if is_pretrained else None)
                m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
            case "efficientnet":
                w = EfficientNet_B0_Weights.IMAGENET1K_V1
                m = efficientnet_b0(weights=w if is_pretrained else None)
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
            case "mini":
                m = MiniSunglassesFeatures()
            case _:
                raise NotImplementedError(f"{model} is not a valid choice!")

        return m
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
    
    def predict(self, img: str | Image.Image) -> str:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")

        transform = T.Compose([T.ToTensor(), T.Normalize([.5] * 3, [.5] * 3)])

        x = transform(img).unsqueeze(0)
        y_hat = self(x).sigmoid().item()

        prediction = "wears sunglasses" if round(y_hat) else "no sunglasses"
        confidence = y_hat if round(y_hat) else 1 - y_hat

        return prediction, confidence


class MiniSunglassesFeatures(nn.Module):
    def __init__(self, weights: str | os.PathLike | None = None):
        super().__init__()

        # MINI V3
        self.features = nn.Sequential(
            self._create_block(3, 5, 3),
            self._create_block(5, 10, 3),
            self._create_block(10, 15, 3),
            self._create_block(15, 20, 3),
            self._create_block(20, 25, 3),
            self._create_block(25, 80, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fc = nn.Linear(80, 1)

        if weights is not None:
            self.load_state_dict(weights)

    def _create_block(self, num_in: int, num_out: int, filter_size: int) \
                     -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(num_in, num_out, filter_size, 1, "valid", bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_out),
            nn.MaxPool2d(2, 2),
        )
    
    def forward(self, x):
        return self.fc(self.features(x))

    
