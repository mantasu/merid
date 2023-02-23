import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from torchmetrics.classification import BinaryF1Score

class MaskGenerator(pl.LightningModule):
    def __init__(self, config={}):
        super().__init__()

        self.domain_adapter = config["domain_adapter"]
        self.glasses_masker = config["glasses_masker"]
        self.shadows_masker = config["shadows_masker"]
        self.discriminator = config.get("discriminator", None)
        
        self.optimizer_g = config.get("optimizer_g", lambda x: optim.Adam(x))
        self.optimizer_d = config.get("optimizer_d", lambda x: optim.Adam(x))
        self.optimizer_m = config.get("optimizer_m", lambda x: optim.Adam(x))

        self.criterion_adv = config.get("criterion_adv", nn.MSELoss())
        self.criterion_seg = config.get("criterion_seg", nn.BCEWithLogitsLoss())

        self.is_da_frozen = all(not p.requires_grad for p in self.domain_adapter.parameters())
        
    def forward(self, x):
        # Get feature vectors for glasses and shadows
        f_glasses, f_shadows = self.domain_adapter(x)

        # Get the glasses output and convert a mask
        out_glasses = self.glasses_masker(f_glasses)
        mask_glasses = out_glasses.argmax(1).unsqueeze(1).float()

        # Update shadow feature and get shadows output for mask
        f_shadows = torch.cat([f_shadows, mask_glasses], dim=1)
        out_shadows = self.shadows_masker(f_shadows)

        return out_glasses, out_shadows
    
    def training_step_domain_adapter(self, batch, train_generator=False):
        # Get real and synthetic images
        real_imgs, synth_imgs = batch

        if train_generator:
            # Concatenate glasses and shadows feature channels together
            f_synth = torch.cat(self.domain_adapter(synth_imgs), axis=1)

            # Get discriminator preds and make labels
            y_hat_synth = self.discriminator(f_synth)
            y_real = torch.ones_like(y_hat_synth, requires_grad=True)

            # Generator loss is the defined adversarial loss
            g_loss = self.criterion_adv(y_hat_synth, y_real)
            self.log('train/g_loss', g_loss)
            
            return g_loss
        
        if not train_generator:
            # Concatenate glasses and shadows feature channels together
            f_real = torch.cat(self.domain_adapter(real_imgs), axis=1)
            f_synth = torch.cat(self.domain_adapter(synth_imgs), axis=1)

            # Get discriminator preds and make labels
            y_hat_real = self.discriminator(f_real.detach())
            y_hat_synth = self.discriminator(f_synth.detach())
            y_real = torch.ones_like(y_hat_real, requires_grad=True)
            y_synth = torch.zeros_like(y_hat_synth, requires_grad=True)

            # Discriminator loss is the mean of real and fake loss
            real_loss = self.criterion_adv(y_hat_real, y_real)
            fake_loss = self.criterion_adv(y_hat_synth, y_synth)
            d_loss = (real_loss + fake_loss) / 2
            self.log('train/d_loss', d_loss)

            return d_loss

    def training_step_mask_segmenter(self, batch):
        # Get the input, labels and preds
        imgs, glasses_y, shadows_y = batch
        out_glasses, out_shadows = self(imgs)

        # Compute glasses & shadows segmentation loss (for one channel)
        loss_glasses = self.criterion_seg(out_glasses[:, 1], glasses_y)
        loss_shadows = self.criterion_seg(out_shadows[:, 1], shadows_y)
        
        # Total loss is just the mean of both, log it
        loss_mask = (loss_glasses + loss_shadows) / 2
        self.log("train/m_loss", loss_mask)

        return loss_mask
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if len(batch) == 4 and optimizer_idx == 0:
            # If batch contains real, synth images and labels
            return self.training_step_mask_segmenter(batch[1:])
        elif len(batch) == 4:
            # If batch contains real, synth images and labels, idx > 0
            return self.training_step_domain_adapter(batch[:2], optimizer_idx==1)
        elif len(batch) == 3:
            # If batch contains only synth images + labels
            return self.training_step_mask_segmenter(batch)
        elif len(batch) == 2:
            # If batch contains only real images and synthetic images
            return self.training_step_domain_adapter(batch, optimizer_idx==0)

    def validation_step_domain_adapter(self, batch):
        # Get real and synthetic images
        real_imgs, synth_imgs = batch

        # Concatenate glasses and shadows feature channels together
        f_real = torch.cat(self.domain_adapter(real_imgs), axis=1)
        f_synth = torch.cat(self.domain_adapter(synth_imgs), axis=1)

        # Get discriminator preds and make labels
        y_hat_real = self.discriminator(f_real)
        y_hat_synth = self.discriminator(f_synth)
        y_real = torch.ones_like(y_hat_real)
        y_synth = torch.zeros_like(y_hat_synth)

        # Generator loss is the defined adversarial loss
        g_loss = self.criterion_adv(y_hat_synth, y_real)
        self.log("val/g_loss", g_loss)

        # Compute discriminator loss for real and synthetic features
        real_loss = self.criterion_adv(y_hat_real, y_real)
        synth_loss = self.criterion_adv(y_hat_synth, y_synth)
        d_loss = (real_loss + synth_loss) / 2
        self.log("val/d_loss", d_loss)

        return {"val/g_loss": g_loss, "val/d_loss": d_loss}
    
    def validation_step_mask_segmenter(self, batch):
        # Get the input, labels and preds
        imgs, y_glasses, y_shadows = batch
        out_glasses, out_shadows = self(imgs)

        # Compute glasses & shadows segmentation loss (for one channel)
        loss_glasses = self.criterion_seg(out_glasses[:, 1], y_glasses)
        loss_shadows = self.criterion_seg(out_shadows[:, 1], y_shadows)

        # Total loss is just the mean of both, log it
        loss_mask = (loss_glasses + loss_shadows) / 2
        self.log("val/m_loss", loss_mask)

        # Get the binary glasses and shadows mask for F1 accuracy
        mask_glasses = out_glasses.argmax(1).unsqueeze(1).float()
        mask_shadows = out_shadows.argmax(1).unsqueeze(1).float()

        # Compute the F1 accuracy and average
        metric = BinaryF1Score().to(self.device)
        f1_glasses = metric(mask_glasses.squeeze(), y_glasses)
        f1_shadows = metric(mask_shadows.squeeze(), y_shadows)
        f1 = (f1_glasses + f1_shadows) / 2
        self.log("val/m_acc", f1)

        return {"val/m_loss": loss_mask, "val/m_acc": f1}
    
    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            # If batch contains real, synth images and labels
            eval = self.validation_step_domain_adapter(batch[:2])
            eval |= self.validation_step_mask_segmenter(batch[1:])
            return eval
        elif len(batch) == 3:
            # If batch contains only synth images + labels
            return self.validation_step_mask_segmenter(batch)
        elif len(batch) == 2:
            # If batch contains only real and synthetic images
            return self.validation_step_domain_adapter(batch)
    
    def configure_optimizers_domain_adapter(self):
        # Get optimizers for discriminator and for generator
        optimizers = [
            self.optimizer_g(self.domain_adapter.parameters()),
            self.optimizer_d(self.discriminator.parameters())
        ]

        return optimizers
    
    def configure_optimizers_mask_segmenter(self):
        # Parameters are from glasses and shadow maskers
        parameters = list(self.glasses_masker.parameters()) +\
                     list(self.shadows_masker.parameters())
        
        if not self.is_da_frozen:
            # Also add parameters of the domain adapter
            parameters += list(self.domain_adapter.parameters())
        
        return self.optimizer_m(parameters)

        
    def configure_optimizers(self):
        optimizers = []

        if self.discriminator is not None:
            # If domain adapter should be trained with LSGAN
            optimizers = self.configure_optimizers_domain_adapter()
        
        optimizers.append(self.configure_optimizers_mask_segmenter())
        
        return optimizers, []