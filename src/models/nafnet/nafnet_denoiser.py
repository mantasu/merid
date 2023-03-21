import sys
import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl

from .nafnet import NAFNet
from torch.optim import AdamW, lr_scheduler

sys.path.append("src")

from data.denoise_data import DenoiseSyntheticDataModule
from utils.training import train, compute_gamma, plot_results
from utils.image_tools import unnormalize


class NAFNetDenoiser(pl.LightningModule):
    def __init__(self, num_epochs=-1, **kwargs):
        # Initialize NAFNet network
        self.nafnet = NAFNet(**kwargs)

        # Initialize some metrics to monitor the performance
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.StructuralSimilarityIndexMeasure(),
            torchmetrics.PeakSignalNoiseRatio()
        ])

        # Loss is just least squares
        self.loss_fn = nn.MSELoss()
        self.num_epochs = num_epochs

    def forward(
        self, 
        img_glasses: torch.Tensor,      # Normalized
        img_inpainted: torch.Tensor,    # Normalized
        mask: torch.Tensor              # Unnormalized, values in full range [0, 1]
    ) -> torch.Tensor:
        
        # Concatenate channel-wise, perform inference, add to grayscale
        grayscale = unnormalize(img_inpainted).mean(dim=1, keepdim=True)
        input = torch.cat((img_glasses, img_inpainted, mask), dim=1)
        grayscale += self.nafnet(input)

        return grayscale
    
    def training_step(self, batch: tuple[torch.Tensor]) -> torch.Tensor:
        # Retrieve elements from the batch: 3 images and 2 masks
        img_glasses, img_inpainted, img_no_glasses, mask_gen, mask_true = batch

        # Compute the value of y_hat and determine pixels
        y_hat = self(img_glasses, img_inpainted, mask_gen)
        is_glassses = mask_true.round().bool()

        # Compute MSE at places where the mask actually exists
        loss = self.loss_fn(y_hat[is_glassses], img_no_glasses[is_glassses])
        self.log("train_loss", loss)
        
        return loss

    def validation_step(self, batch: tuple[torch.Tensor]
                        ) -> dict[str, torch.Tensor]:
        # Retrieve elements from the batch: 3 images and 2 masks
        img_glasses, img_inpainted, img_no_glasses, mask_gen, mask_true = batch

        # Compute the value of y_hat and determine pixels
        y_hat = self(img_glasses, img_inpainted, mask_gen)
        is_glassses = mask_true.round().bool()

        # Compute MSE at places where the mask actually exists
        loss = self.loss_fn(y_hat[is_glassses], img_no_glasses[is_glassses])
        
        return {"loss": loss, "y": img_no_glasses, "y_hat": y_hat}
    
    def validation_step_end(self, outputs: dict[str, torch.Tensor],
                            is_val: bool = True):
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
    
    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, is_val=False)
    
    def configure_optimizers(self):
        # Compute exponential decay rate, set up optimizer and scheduler
        gamma = compute_gamma(self.num_epochs, start_lr=2e-3, end_lr=2e-5)
        optimizer_mse = AdamW(self.parameters(), lr=2e-3, weight_decay=1e-4)
        scheduler_mse = lr_scheduler.ExponentialLR(optimizer_mse, gamma)

        return [optimizer_mse], [scheduler_mse]
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # More efficient optimizer grad zeroing
        optimizer.zero_grad(set_to_none=True)


def run_train(model_name: str = "denoiser", **kwargs):

    model = NAFNetDenoiser(num_epochs=kwargs.get("max_epochs", 10))
    datamodule = DenoiseSyntheticDataModule()

    train(
        model=model,
        datamodule=datamodule,
        model_name=model_name,
        max_epochs=kwargs.get("max_epochs", 10),
        limit_val_batches=kwargs.get("limit_val_batches", 40),
        limit_test_batches=kwargs.get("limit_test_batches", 40),
    )

def plot(weights_path: str = "checkpoints/recolorizer-best.pth"):
    model = NAFNetDenoiser()
    datamodule = DenoiseSyntheticDataModule()

    plot_results(
        model,
        datamodule,
        4,
        weights_path,
        unnormalize=[False, True, False, False],
        is_grayscale=[True, False, False, False]
    )


if __name__ == "__main__":
    run_train()