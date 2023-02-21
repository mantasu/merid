from .unetr_parts import *


class UNetRFull(nn.Module):
    def __init__(self, n_channels, n_classes=1,  args=''):

        # device_count = torch.cuda.device_count()

        # self.cuda0 = torch.device('cuda:0')
        # self.cuda1 = torch.device('cuda:0')
        # self.model_parallelism = model_parallelism
        # if self.model_parallelism:
        #     if device_count > 1:
        #         self.cuda1 = torch.device('cuda:1')
        #         print('Using Model Parallelism with 2 gpu')
        #     else:
        #         print('Can not use model parallelism! Only found 1 GPU device!')
        #         self.cuda1 = torch.device('cuda:0')

        super(UNetRFull, self).__init__()

        # input_feature_len = len(args.input_feature.split(sep=',')) - 1
        # input_feature_len=1
        # if args.use_sagital:
        #     n_channels += 1

        n_filter = [8, 16, 32, 64, 128]

        self.inc = inconv(n_channels, n_filter[0])
        self.down1 = down(n_filter[0], n_filter[1])
        self.down2 = down(n_filter[1], n_filter[2])
        self.down3 = down(n_filter[2], n_filter[3])
        self.down4 = down(n_filter[3], n_filter[3])
        self.up1 = up(n_filter[4], n_filter[2])
        self.up2 = up(n_filter[3], n_filter[1])
        self.up3 = up(n_filter[2], n_filter[0])
        self.up4 = up(n_filter[1], n_filter[0])
        self.outc = outconv(n_filter[0], 3)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)



        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # print(x.shape)
        # x = x.view(-1, 3, 256 , 256)
        # xs = x
        # reg_output = self.net_linear(xs)
        
        return x