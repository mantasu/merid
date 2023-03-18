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
from nafnet.artefact_remover import NAFNetArtefactRemover
from pesr.domain_adaption import PatchGAN
from pytorch_lightning import seed_everything

sys.path.append("src")

from utils.training import get_checkpoint_callback, compute_gamma
from utils.augment import unnormalize
from utils.image_tools import tensor_to_image, image_to_tensor
from utils.io import load_image
from data.nafnet_dataset import NafnetDataModule
from merid.recolorizer import Recolorizer
# from .ddnm.ddnm_inpainter import DDNMInpainter




class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.patch_gan = PatchGAN(in_channels=3, num_filters=32)
        self.classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm2d(1),
            nn.Flatten(),
            nn.Linear(31 * 31, 1),
            nn.Flatten(start_dim=0)
        )

    def forward(self, x):  
        return self.classifier(self.patch_gan(x))
    



class MaskInpainter(pl.LightningModule):
    def __init__(self, config={}, num_epochs=-1):
        super().__init__()
        self.num_epochs = num_epochs
        # self.main_inpainter = config["main_inpainter"]
        #3 self.supp_inpainter: Optional[DDNMInpainter] = config.get("supp_inpainter")
        # self.post_processer = config.get("post_processer")
        # self.discriminator = config.get("discriminator")

        self.post_processer = NAFNetArtefactRemover(**{"nafnet_weights": "checkpoints/NAFNet-SIDD-width32.pth", "da_weights": "checkpoints/pretrained.pt", "width": 32, "middle_blk_num": 12, "enc_blk_nums": [2, 2, 4, 8], "dec_blk_nums": [2, 2, 2, 2]})
        self.recolorizer = Recolorizer()
        self.recolorizer.load_state_dict(torch.load("checkpoints/recolorizer-resnet.pth"))

        # self.discriminator = Discriminator()

        # self.is_ci_frozen = all(not p.requires_grad for p in self.core_inpainter.parameters())

        # Initialize some metrics to monitor the performance
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.StructuralSimilarityIndexMeasure(),
            torchmetrics.PeakSignalNoiseRatio()
        ])

        self.adversarial_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, img, mask, a, b, is_sunglasses):
        img_new = self.main_inpainter(unnormalize(img), mask)
        
        # if self.supp_inpainter is not None:
        #     pass

        # if self.post_processer is not None:
        #     pass

        # inp = normalize(unnormalize(img), [.5] * 3, [.5] * 3)
        # rand = normalize(img_new * 2 - 1, [0, 0, 0], [1, 1, 1])
        # rand = normalize(img_new, [0, 0, 0], [1, 1, 1])
        # rand = img_new * 2 - 1
        # img_new = self.supp_inpainter(inp, 1-mask, None, show_progress=True) # As per official implementation, note: not (x.clip(-1, 1) + 1.0) / 2.0
        # img_new = ((img_new + 1) / 2).clamp(0, 1)

        # if self.post_processer is not None:
        #     feat = self.post_processer(img, mask, feat)
        
        return img_new
    
    def training_step_gan(self, batch, train_generator=False):
        # Get batch
        img_glas, img_no_glass, img_inp, mask_gen = batch
        mask_gen = mask_gen[:, None, ...].round().float()

        if train_generator:
            # X
            x_fake = self.post_processer(img_glas, img_inp, mask_gen)

            # Get discriminator preds and make labels
            y_hat_fake = self.discriminator(x_fake)
            y_real = torch.ones_like(y_hat_fake, requires_grad=True)

            # Generator loss is the defined adversarial loss
            g_loss = self.adversarial_loss(y_hat_fake, y_real)
            self.log("train_loss_g", g_loss, prog_bar=True)
            
            return g_loss
        else:
            # X
            x_real = img_no_glass
            x_fake = self.post_processer(img_glas, img_inp, mask_gen)

            # Get discriminator preds and make labels
            y_hat_real = self.discriminator(x_real)
            y_hat_fake = self.discriminator(x_fake.detach())

            y_real = torch.ones_like(y_hat_real, requires_grad=True)
            y_fake = torch.zeros_like(y_hat_fake, requires_grad=True)

            # Discriminator loss is the mean of real and fake loss
            real_loss = self.adversarial_loss(y_hat_real, y_real)
            fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
            d_loss = (real_loss + fake_loss) / 2
            self.log("train_loss_d", d_loss, prog_bar=True)

            return d_loss
    
    def training_step_mse(self, batch):
        img_glas, img_no_glass, img_inp, mask_gen = batch

        y_hat1 = self.post_processer(img_glas, img_inp, mask_gen)
        # y_hat2 = self.recolorizer(y_hat1, img_glas)
        # y_hat = torch.cat((y_hat1, y_hat2), dim=1)

        y1 = img_no_glass.mean(dim=1, keepdim=True)
        # y2 = img_no_glass
        # y = torch.cat((y1, y2), dim=1)

        # loss = F.mse_loss(y_hat, y)
        
        loss = F.mse_loss(y_hat1[mask_gen.round().bool()], y1[mask_gen.round().bool()])

        # mx = 8
        # print(mask_gen.shape)
        # out_list = [unnormalize(img_glas)[:mx], img_no_glass[:mx], unnormalize(img_inp)[:mx], mask_gen.repeat(1, 3, 1, 1)[:mx].round()]
        # out_list[1][~out_list[-1].bool()] = 0

        # import matplotlib.pyplot as plt
        # from utils.convert import tensor_to_image

        # for i in range(mx):
        #     for j, out in enumerate(out_list):
        #         plt.subplot(mx, len(out_list), i * len(out_list) + j + 1)
        #         plt.imshow(tensor_to_image(out[i]))
        
        # plt.show()

        # return 6

        self.log("train_loss_mse", loss, prog_bar=True)  # log training loss
        return loss
    
    def training_step(self, batch, batch_index, optimizer_idx=0):
        if optimizer_idx == 0:    
            return self.training_step_mse(batch["synthetic"])
        elif optimizer_idx == 1:
            return self.training_step_gan(batch["celeba"], train_generator=True)
        elif optimizer_idx == 2:
            return self.training_step_gan(batch["celeba"], train_generator=False)
        else:
            return None
    
    def validation_step_gan(self, batch):
        # Get batch
        img_glas, img_no_glass, img_inp, mask_gen = batch
        mask_gen = mask_gen[:, None, ...].round().float()

        # X
        x_real = img_no_glass
        x_fake = self.post_processer(img_glas, img_inp, mask_gen)

        # Get discriminator preds and make labels
        y_hat_real = self.discriminator(x_real)
        y_hat_fake = self.discriminator(x_fake)
        y_real = torch.ones_like(y_hat_real)
        y_fake = torch.zeros_like(y_hat_fake)

        # Generator loss is the defined adversarial loss
        g_loss = self.adversarial_loss(y_hat_fake, y_real)
        # self.log("val_g_loss_d", g_loss, prog_bar=True)

        # Compute discriminator loss for real and fake features
        real_loss = self.adversarial_loss(y_hat_real, y_real)
        fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
        d_loss = (real_loss + fake_loss) / 2
        # self.log("val_loss_d", d_loss, prog_bar=True)

        return {"val_loss_g": g_loss, "val_loss_d": d_loss}
    
    def validation_step_mse(self, batch):
        # Get batch
        img_glas, img_no_glass, img_inp, mask_gen = batch

        # Compute mse
        y_hat1 = self.post_processer(img_glas, img_inp, mask_gen)
        # y_hat2 = self.recolorizer(y_hat1, img_glas)
        # y_hat = torch.cat((y_hat1, y_hat2), dim=1)

        y1 = img_no_glass.mean(dim=1, keepdim=True)
        # y2 = img_no_glass
        # y = torch.cat((y1, y2), dim=1)

        # loss = F.mse_loss(y_hat, y)
        
        loss = F.mse_loss(y_hat1[mask_gen.round().bool()], y1[mask_gen.round().bool()])

        return {"val_loss_mse": loss, "y_hat": y_hat1, "y": y1}
        
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
        mask_gen = mask_gen[:, None, ...].round().float()

        # X
        x_real = img_no_glass
        x_fake = self.post_processer(img_glas, img_inp, mask_gen)

        # Get discriminator preds and make labels
        y_hat_real = self.discriminator(x_real)
        y_hat_fake = self.discriminator(x_fake)
        y_real = torch.ones_like(y_hat_real)
        y_fake = torch.zeros_like(y_hat_fake)

        # Generator loss is the defined adversarial loss
        g_loss = self.adversarial_loss(y_hat_fake, y_real)

        # Compute discriminator loss for real and fake features
        real_loss = self.adversarial_loss(y_hat_real, y_real)
        fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
        d_loss = (real_loss + fake_loss) / 2

        return {"loss_g": g_loss, "loss_d": d_loss}
    
    def test_step_mse(self, batch):
        # Get batch
        img_glas, img_no_glass, img_inp, mask_gen = batch

        # Compute mse
        y_hat1 = self.post_processer(img_glas, img_inp, mask_gen)
        # y_hat2 = self.recolorizer(y_hat1, img_glas)
        # y_hat = torch.cat((y_hat1, y_hat2), dim=1)

        y1 = img_no_glass.mean(dim=1, keepdim=True)
        # y2 = img_no_glass
        # y = torch.cat((y1, y2), dim=1)

        # loss = F.mse_loss(y_hat, y)
        
        loss = F.mse_loss(y_hat1[mask_gen.round().bool()], y1[mask_gen.round().bool()])

        return {"test_loss_mse": loss, "y_hat": y_hat1, "y": y1}
        
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
        gamma = compute_gamma(self.num_epochs, start_lr=5e-4, end_lr=5e-5)
        optimizer_mse = torch.optim.AdamW(self.post_processer.parameters(), lr=5e-4, weight_decay=1e-5)
        scheduler_mse = torch.optim.lr_scheduler.ExponentialLR(optimizer_mse, gamma)
        # optimizer_g = torch.optim.AdamW(self.post_processer.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)
        # optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)

        return [optimizer_mse], [scheduler_mse]
    
    # def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
    #     optimizer.zero_grad(set_to_none=True) # better performance




def main():
    NUM_EPOCHS = 10
    BASE_MODEL = "effunetplusplus"
    PATH = "checkpoints/inpainter-" + BASE_MODEL + ".pth"

    seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("medium")

    # Setup model, datamodule and trainer params
    model = MaskInpainter(num_epochs=NUM_EPOCHS)
    datamodule = NafnetDataModule()
    checkpoint_callback = get_checkpoint_callback(BASE_MODEL, "val_loss_mse")
    
    # Initialize the trainer, train it using datamodule and finally test
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, accelerator="gpu", track_grad_norm=2, gradient_clip_val=0.2, val_check_interval=0.1,
                         callbacks=[checkpoint_callback], limit_val_batches=50, limit_test_batches=10)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)

    # Load the best model from saved checkpoints and save its weights
    best_model = MaskInpainter.load_from_checkpoint(checkpoint_callback.best_model_path)
    torch.save(best_model.state_dict(), PATH)


if __name__ == "__main__":
    main()

