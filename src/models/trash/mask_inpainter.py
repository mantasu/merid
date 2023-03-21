import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

import albumentations as A

from albumentations.pytorch import ToTensorV2

import torchmetrics

from typing import Optional

from pytorch_lightning import seed_everything

sys.path.append("src")

from models.pesr.domain_adaption import PatchGAN
from models.nafnet.nafnet import NAFNet
from data.denoise_data import DenoiseDataModule, DenoiseSyntheticDataModule
from utils.image_tools import unnormalize

from utils.training import get_checkpoint_callback, compute_gamma, train
from models.merid.recolorizer import RecolorizerModule
# from .ddnm.ddnm_inpainter import DDNMInpainter

from collections import OrderedDict

class MaskInpainter(pl.LightningModule):
    def __init__(self, config={}, num_epochs=-1):
        super().__init__()
        self.num_epochs = num_epochs

       
        self.denoiser = NAFNet(**{"width": 32, "middle_blk_num": 12, "enc_blk_nums": [2, 2, 4, 8], "dec_blk_nums": [2, 2, 2, 2]})
        self.denoiser.intro = nn.Conv2d(7, 32, kernel_size=3, padding=1)
        self.denoiser.ending = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        weights, custom_w = torch.load("checkpoints/post_processer/pp-inpainter-nafnet.pth"), OrderedDict()
        
        for key in weights.keys():
            if key.startswith("post_processer"):
                custom_w[key[22:]] = weights[key]

        self.denoiser.load_state_dict(custom_w)

        self.recolorizer = RecolorizerModule()
        self.recolorizer.load_state_dict(torch.load("checkpoints/recolorizer-best.pth"))

        # self.discriminator = PatchGAN(in_channels=4, num_filters=64)

        # Initialize some metrics to monitor the performance
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.StructuralSimilarityIndexMeasure(),
            torchmetrics.PeakSignalNoiseRatio()
        ])

        self.loss_fn = nn.MSELoss()
    
    def forward(self, image, mask, image_inpainted=None):

        if image_inpainted is None and self.inpainter is not None:
            # Inpaint the image based on the provided mask
            image_inpainted = self.inpainter(image, mask)
        elif image_inpainted is None:
            # Just mask the values of the image
            image_inpainted = image * (1 - mask)

        if self.denoiser is None:
            return image_inpainted
        
        return None
    
    def training_step_gan(self, batch, train_generator=False):
        # Get batch
        img_glas, img_inp, img_no_glass, mask_gen = batch

        if train_generator:
            # Generate fake x
            grayscale = unnormalize(img_inp).mean(dim=1, keepdim=True)
            grayscale += self.denoiser(torch.cat((img_glas, img_inp, mask_gen), dim=1))
            colorized = self.recolorizer(grayscale, img_glas)
            x_fake = torch.cat((grayscale, colorized), dim=1)

            # Get discriminator preds and make labels
            y_hat_fake = self.discriminator(x_fake)
            y_real = torch.ones_like(y_hat_fake, requires_grad=True)

            # Generator loss is the defined adversarial loss
            g_loss = self.loss_fn(y_hat_fake, y_real)
            self.log("train_loss_g", g_loss, prog_bar=True)
            
            return g_loss
        else:
            # X
            x_real = torch.cat((img_no_glass.mean(dim=1, keepdim=True), img_no_glass), dim=1)
            grayscale = unnormalize(img_inp).mean(dim=1, keepdim=True)
            grayscale += self.denoiser(torch.cat((img_glas, img_inp, mask_gen), dim=1))
            colorized = self.recolorizer(grayscale, img_glas)
            x_fake = torch.cat((grayscale, colorized), dim=1)

            # Get discriminator preds and make labels
            y_hat_real = self.discriminator(x_real)
            y_hat_fake = self.discriminator(x_fake.detach())

            y_real = torch.ones_like(y_hat_real, requires_grad=True)
            y_fake = torch.zeros_like(y_hat_fake, requires_grad=True)

            # Discriminator loss is the mean of real and fake loss
            real_loss = self.loss_fn(y_hat_real, y_real)
            fake_loss = self.loss_fn(y_hat_fake, y_fake)
            d_loss = (real_loss + fake_loss) / 2
            self.log("train_loss_d", d_loss, prog_bar=True)

            return d_loss

    def training_step_mse(self, batch):
        img_glas, img_inp, img_no_glass, mask_gen = batch

        grayscale = unnormalize(img_inp).mean(dim=1, keepdim=True)
        grayscale += self.denoiser(torch.cat((img_glas, img_inp, mask_gen), dim=1))
        colorized = self.recolorizer(grayscale, img_glas)
        
        # y_hat = torch.cat((grayscale, colorized), dim=1)
        # y = torch.cat((img_no_glass.mean(dim=1, keepdim=True), img_no_glass), dim=1)

        mask = mask_gen.round().bool()
        loss1 = self.loss_fn(grayscale[mask], img_no_glass.mean(dim=1, keepdim=True)[mask])
        loss2 = self.loss_fn(colorized, img_no_glass)
        loss = (loss1 + loss2) / 2

        self.log("train_loss_mse", loss, prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_index, optimizer_idx=0):
        if optimizer_idx == 0:    
            return self.training_step_mse(batch)
        elif optimizer_idx == 1:
            return self.training_step_gan(batch["celeba"], train_generator=True)
        elif optimizer_idx == 2:
            return self.training_step_gan(batch["celeba"], train_generator=False)
        else:
            return None
    
    def validation_step_gan(self, batch):
        # Get batch
        img_glas, img_inp, img_no_glass, mask_gen = batch

        # X
        x_real = torch.cat((img_no_glass.mean(dim=1, keepdim=True), img_no_glass), dim=1)
        grayscale = unnormalize(img_inp).mean(dim=1, keepdim=True)
        grayscale += self.denoiser(torch.cat((img_glas, img_inp, mask_gen), dim=1))
        colorized = self.recolorizer(grayscale, img_glas)
        x_fake = torch.cat((grayscale, colorized), dim=1)

        # Get discriminator preds and make labels
        y_hat_real = self.discriminator(x_real)
        y_hat_fake = self.discriminator(x_fake)
        y_real = torch.ones_like(y_hat_real)
        y_fake = torch.zeros_like(y_hat_fake)

        # Generator loss is the defined adversarial loss
        g_loss = self.loss_fn(y_hat_fake, y_real)
        # self.log("val_g_loss_d", g_loss, prog_bar=True)

        # Compute discriminator loss for real and fake features
        real_loss = self.loss_fn(y_hat_real, y_real)
        fake_loss = self.loss_fn(y_hat_fake, y_fake)
        d_loss = (real_loss + fake_loss) / 2
        # self.log("val_loss_d", d_loss, prog_bar=True)

        return {"val_loss_g": g_loss, "val_loss_d": d_loss}
    
    def validation_step_mse(self, batch):
        # Get batch
        img_glas, img_inp, img_no_glass, mask_gen = batch

        grayscale = unnormalize(img_inp).mean(dim=1, keepdim=True)
        grayscale += self.denoiser(torch.cat((img_glas, img_inp, mask_gen), dim=1))
        colorized = self.recolorizer(grayscale, img_glas)

        mask = mask_gen.round().bool()
        loss1 = self.loss_fn(grayscale[mask], img_no_glass.mean(dim=1, keepdim=True)[mask])
        loss2 = self.loss_fn(colorized, img_no_glass)
        loss = (loss1 + loss2) / 2

        y_hat = colorized
        y = img_no_glass


        # import matplotlib.pyplot as plt
        # from utils.image_tools import tensor_to_image

        # print(img_glas.shape, img_inp.shape, img_no_glass.shape, mask_gen.shape, grayscale.shape, colorized.shape)

        # plt.subplot(1, 6, 1)
        # plt.imshow(tensor_to_image(unnormalize(img_glas[0])))
        # plt.subplot(1, 6, 2)
        # plt.imshow(tensor_to_image(unnormalize(img_inp[0])))
        # plt.subplot(1, 6, 3)
        # plt.imshow(tensor_to_image(img_no_glass[0]))
        # plt.subplot(1, 6, 4)
        # plt.imshow(tensor_to_image(mask_gen[0])[..., None].repeat(3, axis=2))
        # plt.subplot(1, 6, 5)
        # plt.imshow(tensor_to_image(grayscale[0])[..., None].repeat(3, axis=2))
        # plt.subplot(1, 6, 6)
        # plt.imshow(tensor_to_image(colorized[0]))
        # plt.show()

        # return a

        loss = self.loss_fn(y_hat, y)

        return {"val_loss_mse": loss, "y_hat": y_hat, "y": y}
        
    def validation_step(self, batch, batch_index, dataloader_index=0):
        if dataloader_index == 0:
            return self.validation_step_mse(batch)
        else:
            return self.validation_step_gan(batch)

    def validation_epoch_end_mse(self, outputs):
        # Concatinate all the computed losses to compute the average
        loss_mean = torch.stack([out["val_loss_mse"] for out in outputs]).mean()
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
    
    def validation_epoch_end_gan(self, outputs):
        # Concatinate all the computed losses to compute the average
        loss_g = torch.stack([out["val_loss_g"] for out in outputs]).mean()
        loss_d = torch.stack([out["val_loss_d"] for out in outputs]).mean()

        self.log("val_loss_g", loss_g, prog_bar=True)
        self.log("val_loss_d", loss_d, prog_bar=True)

    def validation_epoch_end(self, outputs):
        self.validation_epoch_end_mse(outputs)
        # self.validation_epoch_end_gan(outputs[1])
    

    def test_step_gan(self, batch):
        # Get batch
        img_glas, img_no_glass, img_inp, mask_gen = batch

        # X
        x_real = torch.cat((img_no_glass.mean(dim=1, keepdim=True), img_no_glass), dim=1)
        grayscale = unnormalize(img_inp).mean(dim=1, keepdim=True)
        grayscale += self.denoiser(torch.cat((img_glas, img_inp, mask_gen), dim=1))
        colorized = self.recolorizer(grayscale, img_glas)
        x_fake = torch.cat((grayscale, colorized), dim=1)

        # Get discriminator preds and make labels
        y_hat_real = self.discriminator(x_real)
        y_hat_fake = self.discriminator(x_fake)
        y_real = torch.ones_like(y_hat_real)
        y_fake = torch.zeros_like(y_hat_fake)

        # Generator loss is the defined adversarial loss
        g_loss = self.loss_fn(y_hat_fake, y_real)

        # Compute discriminator loss for real and fake features
        real_loss = self.loss_fn(y_hat_real, y_real)
        fake_loss = self.loss_fn(y_hat_fake, y_fake)
        d_loss = (real_loss + fake_loss) / 2

        return {"loss_g": g_loss, "loss_d": d_loss}
    
    def test_step_mse(self, batch):
        # Get batch
        img_glas, img_no_glass, img_inp, mask_gen = batch

        grayscale = unnormalize(img_inp).mean(dim=1, keepdim=True)
        grayscale += self.denoiser(torch.cat((img_glas, img_inp, mask_gen), dim=1))
        colorized = self.recolorizer(grayscale, img_glas)

        mask = mask_gen.round().bool()
        loss1 = self.loss_fn(grayscale[mask], img_no_glass.mean(dim=1, keepdim=True)[mask])
        loss2 = self.loss_fn(colorized, img_no_glass)
        loss = (loss1 + loss2) / 2

        y_hat = colorized
        y = img_no_glass
        
        loss = self.loss_fn(y_hat, y)

        return {"test_loss_mse": loss, "y_hat": y_hat, "y": y}
        
    def test_step(self, batch, batch_index, dataloader_index=0):
        if dataloader_index == 0:
            return self.test_step_mse(batch)
        else:
            return self.test_step_gan(batch)

    def test_epoch_end_mse(self, outputs):
        # Concatinate all the computed losses to compute the average
        loss_mean = torch.stack([out["test_loss_mse"] for out in outputs]).mean()

        # Concatinate y_hats and ys and apply the metrics
        y_hat = torch.cat([out["y_hat"] for out in outputs])
        y = torch.cat([out["y"] for out in outputs])
        metrics = self.metrics(y_hat, y)

        # Log the computed performance (loss and acc)
        self.log("test_loss_mse", loss_mean, prog_bar=True)
        self.log("test_ssim", metrics["StructuralSimilarityIndexMeasure"], prog_bar=True)
        self.log("test_psnr", metrics["PeakSignalNoiseRatio"], prog_bar=True)
    
    def test_epoch_end_gan(self, outputs):
        # Concatinate all the computed losses to compute the average
        loss_g = torch.stack([out["loss_g"] for out in outputs]).mean()
        loss_d = torch.stack([out["loss_d"] for out in outputs]).mean()

        self.log("test_loss_g", loss_g, prog_bar=True)
        self.log("test_loss_d", loss_d, prog_bar=True)

    def test_epoch_end(self, outputs):
        self.test_epoch_end_mse(outputs)
        # self.test_epoch_end_gan(outputs[1])
    

    def configure_optimizers(self):
        gamma = compute_gamma(self.num_epochs, start_lr=5e-4, end_lr=2e-5)
        optimizer_mse = torch.optim.AdamW(list(self.denoiser.parameters()) + list(self.recolorizer.parameters()), lr=5e-4, weight_decay=1e-6)
        scheduler_mse = torch.optim.lr_scheduler.ExponentialLR(optimizer_mse, gamma)
        # optimizer_g = torch.optim.AdamW(list(self.denoiser.parameters()) + list(self.recolorizer.parameters()), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)
        # optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)

        return [optimizer_mse], [scheduler_mse]
    
    # def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
    #     optimizer.zero_grad(set_to_none=True) # better performance


def run_train(model_name: str = "denoiser-recolorer", **kwargs):

    model = MaskInpainter(num_epochs=kwargs.get("max_epochs", 10))
    # datamodule = DenoiseDataModule(batch_size=10, num_workers=4)
    datamodule = DenoiseSyntheticDataModule(batch_size=10, num_workers=4)

    train(
        model=model,
        datamodule=datamodule,
        model_name=model_name,
        max_epochs=kwargs.get("max_epochs", 10),
        limit_val_batches=kwargs.get("limit_val_batches", 40),
        limit_test_batches=kwargs.get("limit_test_batches", 40),
        val_check_interval=0.1,
        monitored_loss_name="val_loss_mse",
        gradient_clip_val=None,
    )

if __name__ == "__main__":
    run_train()

