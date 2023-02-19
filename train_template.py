from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import os
import torchvision
import torchvision.models as models
import segmentation_models_pytorch as smp
import torch
import pytorch_lightning as pl
import glob
model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)



class Synset(Dataset):
    def __init__(self):
        names = glob.glob('./data/synthetic/train_x/img-Glass0[0-9][0-9]*')
        self.X,self.Y=[],[]
        for i in range(len(names)):
                x = torchvision.io.read_image(names[i])
                y = names[i].replace('train_x', 'train_y').replace('-all','-seg')
                y = torchvision.io.read_image(y)
                self.X.append(x)
                self.Y.append(y)
    def __getitem__(self, idx):
        x,y = self.X[idx], self.Y[idx]
        return x.float(),y.float()
    def __len__(self):
        return len(self.X)
synset = Synset()
train_set_size = int(len(synset) * 0.8)
valid_set_size = len(synset) - train_set_size
train_set, valid_set = data.random_split(synset, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=100,num_workers=24)
val_loader = DataLoader(valid_set, batch_size=100, num_workers=24)

x,y = train_set[1]
print(x.shape, y.shape)
# quit()

class RG(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        self.loss = torch.nn.MSELoss()
    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimiser
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out,y)
        return {'loss':loss}
    def train_dataloader(self):
        return train_loader
    
model = RG()
trainer = pl.Trainer(accelerator='gpu', devices=1)
trainer.fit(model)