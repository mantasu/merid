import sys
import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import AdamW, lr_scheduler
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.ops import SqueezeExcitation, Conv2dNormActivation

sys.path.append("src")

from data.colorize_data import RecolorizeDataModule
from utils.training import train, compute_gamma, plot_results


class Recolorizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_ref = nn.Sequential(
            Conv2dNormActivation(3, 12, 3, 1, 1, 1, nn.InstanceNorm2d),
            InvertedResidual(12, 12, 1, 2, nn.InstanceNorm2d),
            SqueezeExcitation(12, 12)
        )

        self.encoder_mid = nn.Sequential(
            Conv2dNormActivation(4, 8, 3, 1, 1, 1, nn.InstanceNorm2d),
            InvertedResidual(8, 8, 1, 2, nn.InstanceNorm2d),
            SqueezeExcitation(8, 8),
            nn.ReLU()
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
    
    def forward(self, grayscale_image, reference_image):
        concat = torch.cat((grayscale_image, reference_image), dim=1)
        enc1 = self.encoder_ref(reference_image)
        enc2 = self.encoder_mid(concat)

        features = self.encoder(torch.cat((concat, enc1), dim=1))
        out = self.decoder(torch.cat((features, enc2), dim=1))

        return (out + 1) / 2


class RecolorizerModule(pl.LightningModule):
    def __init__(self, num_epochs=-1):
        super().__init__()

        # Set up the number of epochs
        self.num_epochs = num_epochs

        # Initialize main recolorizer
        self.recolorizer = Recolorizer()

        # Initialize loss function
        self.loss_fn = nn.MSELoss()

        # Initialize some metrics to monitor the performance
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.StructuralSimilarityIndexMeasure(),
            torchmetrics.PeakSignalNoiseRatio()
        ])
        
    def forward(self, grayscale_image, reference_image):
        return self.recolorizer(grayscale_image, reference_image)

    def training_step(self, batch, batch_idx):
        # Get grayscale, reference & target
        grayscale, reference, target = batch
        
        # Perform forward pass and compute loss
        colorized = self(grayscale, reference)
        loss = self.loss_fn(colorized, target)
        self.log("train_loss", loss)

        return loss
        
    def validation_step(self, batch, batch_idx):
        # Get grayscale, reference & target
        grayscale, reference, target = batch

        # Perform forward pass and compute loss
        colorized = self(grayscale, reference)
        loss = self.loss_fn(colorized, target)

        return {"loss": loss, "y_hat": colorized, "y": target}
    
    def validation_epoch_end(self, outputs, is_val=True):
        # Concatinate all the computed losses to compute the average
        loss_mean = torch.stack([out["loss"] for out in outputs]).mean()
        
        # Concatinate y_hats and ys and apply the metrics
        y_hat = torch.cat([out["y_hat"] for out in outputs])
        y = torch.cat([out["y"] for out in outputs])
        metrics = list(self.metrics(y_hat, y).values())

        if is_val:
            # If it's validation step, also show the learning rate
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log("lr", lr, prog_bar=True)
        
        # Log validation or test MSE, SSIM, and PSNR to the progress bar
        self.log(f"{'val' if is_val else 'test'}_loss", loss_mean, True)
        self.log(f"{'val' if is_val else 'test'}_ssim", metrics[0], True)
        self.log(f"{'val' if is_val else 'test'}_psnr", metrics[1], True)
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    # def test_epoch_end(self, outputs):
    #    return self.validation_epoch_end(outputs, is_val=False)
    
    def configure_optimizers(self):
        # Compute exponential decay rate, set up optimizer and scheduler
        gamma = compute_gamma(self.num_epochs, start_lr=2e-3, end_lr=2e-5)
        optimizer_mse = AdamW(self.parameters(), lr=2e-3, weight_decay=1e-4)
        scheduler_mse = lr_scheduler.ExponentialLR(optimizer_mse, gamma)

        return [optimizer_mse], [scheduler_mse]
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)


def run_train(model_name: str = "recolorizer", **kwargs):

    model = RecolorizerModule(num_epochs=kwargs.get("max_epochs", 20))
    datamodule = RecolorizeDataModule()

    train(
        model=model,
        datamodule=datamodule,
        model_name=model_name,
        max_epochs=kwargs.get("max_epochs", 20),
        limit_val_batches=kwargs.get("limit_val_batches", 50),
        limit_test_batches=kwargs.get("limit_test_batches", 50),
        val_check_interval=0.1,
    )

def run_test():
    model = RecolorizerModule().load_from_checkpoint("checkpoints/unused/recolorizer-epoch=18-val_loss=0.00005.ckpt")
    datamodule = RecolorizeDataModule()

    trainer = pl.Trainer(accelerator="gpu")
    trainer.test(model, datamodule=datamodule)

def plot(weights_path: str = "checkpoints/recolorizer-best.pth"):
    model = RecolorizerModule()
    datamodule = RecolorizeDataModule()

    plot_results(
        model,
        datamodule,
        2,
        weights_path,
        unnormalize=[False, True, False, False],
        is_grayscale=[True, False, False, False]
    )


if __name__ == "__main__":
    run_test()