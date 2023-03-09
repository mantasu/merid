import torch
import torch.nn as nn

from .nafnet import NAFNet
from torchvision.ops import SqueezeExcitation
from torchvision.ops.misc import Conv2dNormActivation

class InvertedResidualSEReduction(nn.Module):
    def __init__(self, in_channels: int, kernel_size_expand: int = 3,
                 kernel_size_reduce: int = 3, expand_ratio: float = 2,
                 use_final_bias: bool = True):
        super().__init__()
        hidden_dim = round(in_channels * expand_ratio)

        self.expansion_conv = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size_expand,
            norm_layer=nn.InstanceNorm2d,
            activation_layer=nn.SiLU
        )

        self.depthwise_conv = Conv2dNormActivation(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            group=hidden_dim,
            norm_layer=nn.InstanceNorm2d,
            activation_layer=nn.SiLU
        )

        self.squeeze_excitation = SqueezeExcitation(hidden_dim, hidden_dim)

        self.reduction = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=1,
            kernel_size=kernel_size_reduce,
            bias=use_final_bias,
        )

    def forward(self, x):
        x_expanded = self.expansion_conv(x)
        deep_feats = self.depthwise_conv(x_expanded)
        x_residual = x_expanded + self.squeeze_excitation(deep_feats)
        out = self.reduction(x_residual)

        return out


class NAFNetArtefactRemover(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        _kwargs = kwargs.copy()
        nafnet_weights = _kwargs.pop("nafnet_weights", None)

        self.img2chan = self._make_to_chan_block(kwargs.pop("img_in", 3))
        self.msk2chan = self._make_to_chan_block(kwargs.pop("msk_in", 4))
        self.inp2chan = self._make_to_chan_block(kwargs.pop("inp_in", 3))
        self.channel_joiner = nn.Sequential(nn.LayerNorm(), nn.LeakyReLU())
        self.nafnet = NAFNet(**_kwargs)

        if nafnet_weights is not None:
            self.nafnet.load_state_dict(torch.load(nafnet_weights)["params"])
    
    def _make_to_chan_block(self, in_channels):
        return InvertedResidualSEReduction(
            in_channels=in_channels,
            kernel_size_expand=7,
            kernel_size_reduce=7,
            expand_ratio=2,
            use_final_bias=False
        )
    
    def forward(self, img_original, mask, img_inpainted):
        chan1 = self.img2chan(img_original)
        chan2 = self.msk2chan(mask)
        chan3 = self.inp2chan(img_inpainted)
        chans = torch.cat((chan1, chan2, chan3), dim=1)
        out = self.nafnet(self.channel_joiner(chans))

        return out