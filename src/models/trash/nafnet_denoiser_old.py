import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nafnet.nafnet import NAFNet
from torchvision.ops import SqueezeExcitation
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models.segmentation import LRASPP, lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
from torchvision.models.segmentation.lraspp import LRASPPHead

sys.path.append("src")
from utils.guest import fix_pesr_da
from models.pesr.domain_adaption import DomainAdapter

from torchvision.models.mobilenetv2 import InvertedResidual
# from segmentation_models_pytorch.efficientunetplusplus.model import EfficientUn
from segmentation_models_pytorch import EfficientUnetPlusPlus, UnetPlusPlus, ResUnetPlusPlus
from segmentation_models_pytorch.unetplusplus.decoder import UnetPlusPlusDecoder

sys.path.append("src")

from utils.training import get_checkpoint_callback, compute_gamma

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
            groups=hidden_dim,
            norm_layer=nn.InstanceNorm2d,
            activation_layer=nn.SiLU
        )

        self.squeeze_excitation = SqueezeExcitation(hidden_dim, hidden_dim)

        self.reduction = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=1,
            kernel_size=kernel_size_reduce,
            bias=use_final_bias,
            padding="same",
            padding_mode="reflect"
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
        da_weights = _kwargs.pop("da_weights", None)

        # self.da = DomainAdapter(vgg_weights=None, is_torchvision_vgg=False)

        # if da_weights is not None:
        #     self.da.load_state_dict(fix_pesr_da(torch.load(da_weights)["DA"]))
        #     self.da.eval()
        #     for param in self.da.parameters():
        #         param.requires_grad = False
        
        # self.to_two = nn.Conv2d(128, 2, 3, 1, "same", padding_mode="reflect")

        # self.img2chan = self._make_to_chan_block(kwargs.pop("img_in", 3))
        # self.msk2chan = self._make_to_chan_block(kwargs.pop("msk_in", 3))
        # self.inp2chan = self._make_to_chan_block(kwargs.pop("inp_in", 3))
        # self.channel_joiner = nn.Sequential(nn.BatchNorm2d(3), nn.LeakyReLU())

        # w = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        # m = lraspp_mobilenet_v3_large(weights=w)

        # m.backbone["0"] = nn.Sequential(
        #     nn.Conv2d(7, 16, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        #     nn.Hardswish()
        # )
        # m.classifier = LRASPPHead(40, 960, 1, 128)
        # self.lraspp = m

        # self.resunet = ResUnetPlusPlus(in_channels=7, decoder_attention_type="se")
        
        # self.effunet = UnetPlusPlus(encoder_name="resnet50", in_channels=3)
        # 

        # VERSION 1
        # self.unet = UnetPlusPlus(encoder_name="timm-tf_efficientnet_lite0", in_channels=7, classes=1)

        # VERSION 2
        # self.nafnet = NAFNet(**{"width": 32, "middle_blk_num": 12, "enc_blk_nums": [2, 2, 4, 8], "dec_blk_nums": [2, 2, 2, 2]})

        # if nafnet_weights is not None:
        #     self.nafnet.load_state_dict(torch.load(nafnet_weights)["params"])
        
        # self.nafnet.intro = nn.Conv2d(7, 32, kernel_size=3, padding=1)
        # self.nafnet.ending = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        # VERSION 3
        self.effunet = EfficientUnetPlusPlus(in_channels=7)
    
    def _make_to_chan_block(self, in_channels):
        return InvertedResidualSEReduction(
            in_channels=in_channels,
            kernel_size_expand=7,
            kernel_size_reduce=7,
            expand_ratio=2,
            use_final_bias=False
        )
    
    def forward(self, img_original, img_inpainted, mask):

        # chans2 = torch.cat((img_original, mask), dim=1)
        chans3 = torch.cat((img_original, img_inpainted, mask), dim=1)
        # enc1 = self.encoder_ref(chans2)
        # enc2 = self.encoder_mid(chans3)

        # features = self.encoder(torch.cat((chans3, enc1), dim=1))
        # out = self.decoder(torch.cat((features, enc2), dim=1))


        # out = self.effunet(torch.cat((img_original, img_inpainted, mask), dim=1))
        # out = (torch.tanh(out) + 1) / 2
        
        # img_features = torch.cat((self.to_two(torch.cat((*self.da(img_original),), dim=1)), mask), dim=1)
        # chan1 = self.img2chan(img_original)
        # chan2 = self.msk2chan(img_features)
        # chan3 = self.inp2chan(img_inpainted)
        # chans = torch.cat((chan1, chan2, chan3), dim=1)
        # chans = self.channel_joiner(chans)

        # single = self.first(torch.cat((img_original, img_inpainted, mask), dim=1))

        # out = (self.effunet(chans).tanh() + 1) / 2

        out = unnormalize(img_inpainted).mean(dim=1, keepdim=True)
        out[mask.round().bool()] += self.effunet(chans3)[mask.round().bool()]
        
        # out[mask.round().bool()] += [mask.round().bool()]
        
        # out = self.nafnet(single, orig=unnormalize(img_inpainted).mean(dim=1, keepdim=True))
        # ret = unnormalize(img_inpainted).mean(dim=1, keepdim=True)
        # ret[mask.round().bool()] += out[mask.round().bool()].tanh()
        
        
        # out = (torch.tanh(out) + 1) / 2

        # out = self.nafnet(torch.cat((unnormalize(img_original), unnormalize(img_inpainted), mask), dim=1))[:, 2:5].mean(dim=1, keepdim=True)


        # out = self.nafnet(torch.cat((img_original.mean(dim=1, keepdim=True), img_inpainted.mean(dim=1, keepdim=True), mask), dim=1))
        # out = torch.tanh(self.nafnet(torch.cat((img_original, img_inpainted, mask), dim=1))[:, 2:5].mean(dim=1, keepdim=True))

        return out