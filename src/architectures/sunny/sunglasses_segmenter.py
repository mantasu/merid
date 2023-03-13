import sys
import torch
import numpy as np
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl

from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.ops import SqueezeExcitation
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.transforms.functional import to_tensor, normalize, to_pil_image
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.lraspp import LRASPPHead

from torchvision.models.segmentation import (
    deeplabv3_resnet50, DeepLabV3_ResNet50_Weights,
    fcn_resnet50, FCN_ResNet50_Weights,
    lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
)

sys.path.append("src")
from datasets.celeba_mask_hq_dataset import CelebaMaksHQModule
from utils.training import compute_gamma, get_checkpoint_callback

class GlassesSegmenter(pl.LightningModule):
    def __init__(self, num_epochs: int = -1,
                 base_model: str = "deeplab",
                 is_base_pretrained: bool = False):
        super().__init__()
        self.num_epochs = num_epochs
        
        # Load and replace the last layer with a binary segmentation head
        self.base_model = self.load_base_model(base_model, is_base_pretrained)

        # Initialize some metrics to monitor the performance
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.F1Score(task="binary"),
            torchmetrics.Dice()
        ])

        # Define a binary cross-entropy loss function + a scheduling param
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]))
    
    def load_base_model(self, model_name: str, is_pretrained: bool):
        match model_name:
            case "deeplab":
                w = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                m = deeplabv3_resnet50(weights=w if is_pretrained else None)
                m.classifier = DeepLabHead(2048, 1)
                m.aux_classifier = None
            case "fcn":
                w = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                m = fcn_resnet50(weights=w if is_pretrained else None)
                m.classifier[-1] = nn.Conv2d(256, 1, 1)
            case "lraspp":
                w = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
                m = lraspp_mobilenet_v3_large(weights=w if is_pretrained else None)
                m.classifier = LRASPPHead(40, 960, 1, 128)
            case "mini":
                m = MiniSunglassesSegmenter()
            case _:
                raise NotImplementedError(f"{model_name} is not a valid choice!")
        
        return m

    def forward(self, x):
        # Pass the input through the segmentation model
        out = self.base_model(x)["out"]

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
        loss = self.loss_fn(self(x).squeeze(1), y)

        # Log mini-batch train loss
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int) -> dict[str, torch.Tensor]:
        # Get samples and predict
        x, y = batch
        y_hat = self(x).squeeze(1)

        # Compute the mini-batch loss
        loss = self.loss_fn(y_hat, y)

        return {"loss": loss, "y_hat": y_hat, "y": y}
    
    def validation_epoch_end(self, outputs: dict[str, torch.Tensor]):
        # Concatinate all the computed losses to compute the average
        loss_mean = torch.stack([out["loss"] for out in outputs]).mean()
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        # Concatinate y_hats and ys and apply the metrics
        y_hat = torch.cat([out["y_hat"] for out in outputs])
        y = torch.cat([out["y"] for out in outputs])
        metrics = self.metrics(y_hat, y.long())

        # Log the computed performance (loss and acc)
        self.log("val_loss", loss_mean, prog_bar=True)
        self.log("lr", lr, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)

    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor],
                  batch_idx: int) -> dict[str, torch.Tensor]:
        # Get samples and predict
        x, y = batch
        y_hat = self(x).squeeze(1)

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
        gamma = compute_gamma(self.num_epochs, start_lr=1e-3, end_lr=1e-6)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-8)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2, 1e-6)

        return [optimizer], [scheduler]


class MiniSunglassesSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, padding="same")
        self.irb1 = InvertedResidual(256, 256, 2, expand_ratio=6)
        self.irb2 = InvertedResidual(256, 512, 2, expand_ratio=6)
        self.irb3 = InvertedResidual(512, 1024, 2, expand_ratio=6)
        self.se1 = SqueezeExcitation(1024, 4)
        self.conv2 = nn.Conv2d(1024, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.irb1(x)
        x = self.irb2(x)
        x = self.irb3(x)
        x = self.se1(x)
        x = self.conv2(x)
        return x


def main():
    NUM_EPOCHS = 310
    BASE_MODEL = "lraspp"
    LOAD_BASE_WEIGHTS = True
    PATH = "checkpoints/sunglasses-segmenter-" + BASE_MODEL + ".pth"

    seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("medium")

    # Setup model, datamodule and trainer params
    model = GlassesSegmenter(NUM_EPOCHS, BASE_MODEL, LOAD_BASE_WEIGHTS)
    datamodule = CelebaMaksHQModule()
    checkpoint_callback = get_checkpoint_callback(BASE_MODEL)
    
    # Initialize the trainer, train it using datamodule and finally test
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, accelerator="gpu",
                         callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=datamodule, ckpt_path="checkpoints/lraspp-epoch=126.ckpt")
    trainer.test(ckpt_path="best", datamodule=datamodule)

    # Load the best model from saved checkpoints and save its weights
    best_model = GlassesSegmenter.load_from_checkpoint(checkpoint_callback.best_model_path, base_model=BASE_MODEL, is_base_pretrained=False)
    # best_model = SunglassesSegmenter.load_from_checkpoint("checkpoints/deeplab-epoch=00.ckpt", base_model=BASE_MODEL, is_base_pretrained=False)
    torch.save(best_model.state_dict(), PATH)

if __name__ == "__main__":
    main()
