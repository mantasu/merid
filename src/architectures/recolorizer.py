import sys
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning import seed_everything
from torchvision.models import resnet18, ResNet18_Weights

from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.models.segmentation import LRASPP, lraspp_mobilenet_v3_large
from torchvision.models.segmentation.lraspp import LRASPPHead

sys.path.append("src")

from utils.training import get_checkpoint_callback, compute_gamma
from datasets.colorize_dataset import ColorizeDataModule
from architectures.pesr.blocks import ResNetBlock


class Recolorizer(pl.LightningModule):
    def __init__(self, num_epochs=-1):
        super().__init__()
        self.num_epochs = num_epochs

        self.encoder_ref = nn.Sequential(
            InvertedResidual(3, 12, 1, 2, norm_layer=nn.InstanceNorm2d),
            nn.LeakyReLU(),
            nn.Conv2d(12, 12, 3, 1, 1, groups=12),
        )

        self.encoder_mid = nn.Sequential(
            InvertedResidual(4, 8, 1, 2, norm_layer=nn.InstanceNorm2d),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, 3, 1, 1, groups=8),
            nn.ReLU(),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(12 + 4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 56, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(56 + 8, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.loss_fn = nn.MSELoss()

        # Initialize some metrics to monitor the performance
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.StructuralSimilarityIndexMeasure(),
            torchmetrics.PeakSignalNoiseRatio()
        ])
        
    def forward(self, grayscale_image, reference_image):
        concat = torch.cat((grayscale_image, reference_image), dim=1)
        enc1 = self.encoder_ref(reference_image)
        enc2 = self.encoder_mid(concat)

        features = self.encoder(torch.cat((concat, enc1), dim=1))
        out = self.decoder(torch.cat((features, enc2), dim=1))

        return (out + 1) / 2

    def training_step(self, batch, batch_idx):
        # training loop
        ref, target = batch
        ref, target = ref.cuda(non_blocking=True), target.cuda(non_blocking=True)
        
        gray = target.mean(dim=1, keepdim=True)

        colorized = self(gray, ref)
        loss = self.loss_fn(colorized, target)
        self.log("train_loss", loss, prog_bar=True)

        return loss
        
    def validation_step(self, batch, batch_idx):
        # validation loop
        ref, target = batch
        ref, target = ref.cuda(non_blocking=True), target.cuda(non_blocking=True)

        gray = target.mean(dim=1, keepdim=True)
        colorized = self(gray, ref)
        loss = self.loss_fn(colorized, target)
        self.log('val_loss', loss)

        metrics = self.metrics(colorized, target)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True)
        self.log("val_ssim", metrics["StructuralSimilarityIndexMeasure"], prog_bar=True)
        self.log("val_psnr", metrics["PeakSignalNoiseRatio"], prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        # validation loop
        ref, target = batch
        gray = target.mean(dim=1, keepdim=True)
        colorized = self(gray, ref)
        loss = self.loss_fn(colorized, target)
        self.log('val_loss', loss)

        return {"val_loss": loss, "y_hat": colorized, "y": target}
    
    def validation_epoch_end(self, outputs):
        # Concatinate all the computed losses to compute the average
        loss_mean = torch.stack([out["val_loss"] for out in outputs]).mean()
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        # Concatinate y_hats and ys and apply the metrics
        y_hat = torch.cat([out["y_hat"] for out in outputs])
        y = torch.cat([out["y"] for out in outputs])
        metrics = self.metrics(y_hat, y)

        # Log the computed performance (loss and acc)
        self.log("val_loss_mse", loss_mean, prog_bar=True)
        self.log("lr", lr, prog_bar=True)
        self.log("val_ssim", metrics["StructuralSimilarityIndexMeasure"], prog_bar=True)
        self.log("val_psnr", metrics["PeakSignalNoiseRatio"], prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        # test loop
        ref, target = batch
        gray = target.mean(dim=1, keepdim=True)
        colorized = self(gray, ref)
        loss = self.loss_fn(colorized, target)
        self.log('test_loss', loss)

        return {"test_loss": loss, "y_hat": colorized, "y": target}
    
    def test_epoch_end(self, outputs):
        # Concatinate all the computed losses to compute the average
        loss_mean = torch.stack([out["test_loss"] for out in outputs]).mean()
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        # Concatinate y_hats and ys and apply the metrics
        y_hat = torch.cat([out["y_hat"] for out in outputs])
        y = torch.cat([out["y"] for out in outputs])
        metrics = self.metrics(y_hat, y)

        # Log the computed performance (loss and acc)
        self.log("test_loss_mse", loss_mean, prog_bar=True)
        self.log("lr", lr, prog_bar=True)
        self.log("test_ssim", metrics["StructuralSimilarityIndexMeasure"], prog_bar=True)
        self.log("test_psnr", metrics["PeakSignalNoiseRatio"], prog_bar=True)
    
    def configure_optimizers(self):
        gamma = compute_gamma(self.num_epochs, start_lr=2e-3, end_lr=2e-5)
        optimizer_mse = torch.optim.AdamW(self.parameters(), lr=2e-3, weight_decay=1e-3)
        scheduler_mse = torch.optim.lr_scheduler.ExponentialLR(optimizer_mse, gamma)
        # optimizer_g = torch.optim.AdamW(self.post_processer.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)
        # optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)

        return [optimizer_mse], [scheduler_mse]
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True) # better performance


def main():
    NUM_EPOCHS = 5
    BASE_MODEL = "resnet"
    PATH = "checkpoints/recolorizer-" + BASE_MODEL + ".pth"

    seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("medium")

    # Setup model, datamodule and trainer params
    model = Recolorizer(num_epochs=NUM_EPOCHS)
    datamodule = ColorizeDataModule()
    checkpoint_callback = get_checkpoint_callback(BASE_MODEL, monitor="val_loss")
    
    # Initialize the trainer, train it using datamodule and finally test
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, accelerator="gpu", limit_val_batches=50, limit_test_batches=50, gradient_clip_val=0.2,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)

    # Load the best model from saved checkpoints and save its weights
    best_model = Recolorizer.load_from_checkpoint(checkpoint_callback.best_model_path)
    torch.save(best_model.state_dict(), PATH)


if __name__ == "__main__":
    main()