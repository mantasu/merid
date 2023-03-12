import sys
import torch
import numpy as np
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl

from PIL import Image
from torchvision.ops import SqueezeExcitation
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.transforms.functional import to_tensor, normalize, to_pil_image
from torchvision.models.segmentation import (
    deeplabv3_resnet50, DeepLabV3_ResNet50_Weights,
    fcn_resnet50, FCN_ResNet50_Weights,
    lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
)

sys.path.append("src")
from datasets.celeba_mask_hq_dataset import CelebaMaksHQModule
from utils.training import compute_gamma, seed, get_checkpoint_callback

class SunglassesSegmenter(pl.LightningModule):
    def __init__(self, num_epochs: int = -1,
                 deeplab_weights: DeepLabV3_ResNet50_Weights | None = None):
        super().__init__()
        
        # Load and replace the last layer with a binary segmentation head
        self.segmentation_model = deeplabv3_resnet50(weights=deeplab_weights)
        self.segmentation_model.classifier[-1] = nn.Conv2d(256, 1, 1)

        # Initialize some metrics to monitor the performance
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.F1Score(task="binary"),
            torchmetrics.Dice()
        ])

        # Define a binary cross-entropy loss function + a scheduling param
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]))
        self.gamma = compute_gamma(num_epochs, start_lr=1e-3, end_lr=3e-4)
    
    def create_base_model(self, model_name: str, is_pretrained: bool):
        match model_name:
            case "deeplab":
                w = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                m = deeplabv3_resnet50(weights=w if is_pretrained else None)
                m.segmentation_model.classifier[-1] = nn.Conv2d(256, 1, 1)
            case "fcn":
                w = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                m = fcn_resnet50(weights=w if is_pretrained else None)
                m.segmentation_model.classifier[-1] = nn.Conv2d(256, 1, 1)
            case "lraspp":
                w = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
                m = lraspp_mobilenet_v3_large(weights=w if is_pretrained else None)
                m.segmentation_model.classifier[-1] = nn.Conv2d(256, 1, 1)
            case "mini":
                m = MiniSunglassesSegmenter()
            case _:
                raise NotImplementedError(f"{model_name} is not a valid choice!")

    def forward(self, x):
        # Pass the input through the segmentation model
        out = self.segmentation_model(x)["out"].squeeze()

        return out
    
    @torch.no_grad()
    def predict(self, x: Image.Image | np.ndarray):
        # Image to tensor
        x = to_tensor(x)

        # Normalize to standard augmentation values if not already norm
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Compute the mask prediction, round off
        x = self(x.unsqueeze(0)).squeeze().round()

        return to_pil_image(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        # Mini-batch
        x, y = batch

        # Forward pass + loss computation
        loss = self.loss_fn(self(x), y)

        # Log mini-batch train loss
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int) -> dict[str, torch.Tensor]:
        # Get samples and predict
        x, y = batch
        y_hat = self(x)

        # Compute the mini-batch loss
        loss = self.loss_fn(y_hat, y)

        return {"loss": loss, "y_hat": y_hat, "y": y}
    
    def validation_epoch_end(self, outputs: dict[str, torch.Tensor]):
        # Concatinate all the computed losses to compute the average
        loss_mean = torch.stack([out["loss"] for out in outputs]).mean()

        # Concatinate y_hats and ys and apply the metrics
        y_hat = torch.cat([out["y_hat"] for out in outputs])
        y = torch.cat([out["y"] for out in outputs])
        metrics = self.metrics(y_hat, y.long())

        # Log the computed performance (loss and acc)
        self.log("val_loss", loss_mean, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)

    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor],
                  batch_idx: int) -> dict[str, torch.Tensor]:
        # Get samples and predict
        x, y = batch
        y_hat = self(x)

        # Compute the mini-batch loss
        loss = self.loss_fn(y_hat, y)

        return {"loss": loss, "y_hat": y_hat, "y": y}
    
    def test_epoch_end(self, outputs: dict[str, torch.Tensor]):
        # Concatinate all the computed losses to compute the average
        loss_mean = torch.stack([out["loss"] for out in outputs]).mean()

        # Concatinate y_hats and ys and apply the metrics
        y_hat = torch.cat([out["y_hat"] for out in outputs])
        y = torch.cat([out["y"] for out in outputs])
        metrics = self.metrics(y_hat, y.long())

        # Log the computed performance (loss and acc)
        self.log("test_loss", loss_mean, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.gamma)

        return [optimizer], [scheduler]


class MiniSunglassesSegmenter(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1280, 256, kernel_size=1)
        self.irb1 = InvertedResidual(256, 256, 2, expand_ratio=6)
        self.irb2 = InvertedResidual(256, 512, 2, expand_ratio=6)
        self.irb3 = InvertedResidual(512, 1024, 2, expand_ratio=6)
        self.se1 = SqueezeExcitation(1024, 4)
        self.conv2 = nn.Conv2d(1024, 1, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1(x)
        x = self.irb1(x)
        x = self.irb2(x)
        x = self.irb3(x)
        x = self.se1(x)
        x = self.conv2(x)
        return x


def main():
    torch.set_float32_matmul_precision("medium")
    NUM_EPOCHS = 80
    seed(0)

    # Setup model, datamodule and trainer params
    model = SunglassesSegmenter(NUM_EPOCHS, deeplab_weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    datamodule = CelebaMaksHQModule()
    checkpoint_callback = get_checkpoint_callback("mask-segmenter")
    
    # Initialize the trainer, train it using datamodule and finally test
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, accelerator="gpu",
                         callbacks=[checkpoint_callback], resume_from_checkpoint="checkpoints/mask-segmenter-epoch=39.ckpt")
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)

    # Load the best model from saved checkpoints and save its weights
    best_model = SunglassesSegmenter.load_from_checkpoint(checkpoint_callback.best_model_path)
    torch.save(best_model.state_dict(), "checkpoints/sunglasses-segmenter-best.pth")

if __name__ == "__main__":
    main()
