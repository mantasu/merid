from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import os
import torchvision
import torchvision.models as models
import segmentation_models_pytorch as smp
import torch
import pytorch_lightning as pl
import glob
torch.set_float32_matmul_precision('medium')
from torchsummary import summary
import torchvision.transforms as T
# from pytorch_lightning.loggers import TensorBoardLogger
from unet import Unet
# model = Unet.UNetRFull(n_channels=3, n_classes=3)
# print(model(torch.ones((1,3,256,256)).float().cuda()).shape)
# # summary(model,(3,256,256))
# quit()

class Synset(Dataset):
    def __init__(self):
        names = glob.glob('./data/synthetic/train_x/img-Glass0[0-9][0-9]*')
        # names = names[:100]
        self.X,self.Y=[],[]
        for i in range(len(names)):
                x = torchvision.io.read_image(names[i])
                # y = names[i].replace('train_x', 'train_y').replace('-all','-seg')
                y = names[i].replace('train_x', 'train_y').replace('-all','-face')
                y = torchvision.io.read_image(y)
                self.X.append(x)
                self.Y.append(y)
        self.trans = T.Compose([
                # T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                ])
    def __getitem__(self, idx):
        x,y = self.X[idx].float(), self.Y[idx].float()
        x = self.trans(x)
        y = self.trans(y)
        return x,y
    def __len__(self):
        return len(self.X)
synset = Synset()
train_set_size = int(len(synset) * 0.8)
valid_set_size = len(synset) - train_set_size
train_set, valid_set = data.random_split(synset, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=100,num_workers=5+5)
val_loader = DataLoader(valid_set, batch_size=100, num_workers=5+5)

x,y = train_set[1]
print(x.shape, y.shape)
# quit()

class RG(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        # self.model = smp.Unet(
        #     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        #     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #     classes=3,                      # model output channels (number of classes in your dataset)
        # )
        self.model = Unet.UNetRFull(n_channels=3, n_classes=3)
        self.loss = torch.nn.MSELoss()
        # self.logger = TensorBoardLogger("tb_logs", name="my_model")
    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimiser
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out,y)
        self.log("train_loss", loss)
        return {'loss':loss}
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(y, out)
        self.log("val_loss", loss)
        if batch_idx % 10: # Log every 10 batches
            grid = torchvision.utils.make_grid(out)
            self.logger.experiment.add_image('generated_images', grid, global_step=trainer.global_step)
        return {'loss':loss}
    def train_dataloader(self):
        return train_loader
    def val_dataloader(self):
        return val_loader
    
model = RG()
trainer = pl.Trainer(accelerator='gpu', devices=1)
trainer.fit(model)