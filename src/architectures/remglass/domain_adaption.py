import torch.nn as nn
from .blocks import ResNetBlock, VGGBlock

class DomainAdapter(nn.Module):
    def __init__(self, vgg_weights="DEFAULT", is_torchvision_vgg=True):
        super().__init__()
        
        # VGG encoding and 6 residual blocks (3 glass and 3 shadow)
        self.vgg_encoding = VGGBlock(vgg_weights, is_torchvision_vgg)
        self.glass_module = self.build_res_layers()
        self.shadow_module = self.build_res_layers()
    
    def build_res_layers(self):
        # 3 blocks and normalization
        res_layers = nn.Sequential(*[
            ResNetBlock(64),
            ResNetBlock(64),
            ResNetBlock(64),
            nn.InstanceNorm2d(64),
        ])

        return res_layers

    def forward(self, x):
        # Encode and get 2 outputs
        enc = self.vgg_encoding(x)
        glass_out = self.glass_module(enc)
        shadow_out = self.shadow_module(enc)

        return glass_out, shadow_out

class PatchGAN(nn.Module):
    """Generated by ChatGPT (PatchGAN discriminator)"""

    def __init__(self, in_channels=128, num_filters=64, num_layers=3):
        super().__init__()
        
        layers = []

        # First layer
        layers.append(nn.Conv2d(in_channels, num_filters, 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers
        for i in range(1, num_layers):
            layers.append(nn.Conv2d(num_filters*2**(i-1), num_filters*2**i, 4, 2, 1))
            layers.append(nn.InstanceNorm2d(num_filters * 2**i))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
        # Last layer
        layers.append(nn.Conv2d(num_filters * 2**(num_layers-1), 1, 4, 1, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
