import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DUpsampling(nn.Module):
    """DUsampling module"""

    def __init__(self, in_channels, out_channels, scale_factor=2, **kwargs):
        super(DUpsampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv_w = nn.Conv2d(in_channels, out_channels * scale_factor * scale_factor, 1, bias=False)

    def forward(self, x):
        x = self.conv_w(x)
        n, c, h, w = x.size()

        # N, C, H, W --> N, W, H, C
        x = x.permute(0, 3, 2, 1).contiguous()

        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, h * self.scale_factor, c // self.scale_factor)

        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()

        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, h * self.scale_factor, w * self.scale_factor, c // (self.scale_factor * self.scale_factor))

        # N, H * scale, W * scale, C // (scale ** 2) -- > N, C // (scale ** 2), H * scale, W * scale
        x = x.permute(0, 3, 1, 2)

        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=512,
                 out_channels=256,
                 kernel_size=3,
                 is_deconv=False,
                 ):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels // 2)
        self.mish1 = nn.Mish(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        '''
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                              in_channels // 4,
                                              3,
                                              stride=2,
                                              padding=1,
                                              output_padding=conv_padding,bias=False)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2,**up_kwargs)
        '''
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels // 2)
        self.mish2 = nn.Mish(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.mish3 = nn.Mish(inplace=True)

    def forward(self, x):
        x = torch.cat(x, 1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.mish1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.mish2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.mish3(x)
        return x


class ResNet50UnetPlus(nn.Module):
    def __init__(self,
                 num_class,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 ):
        super().__init__()

        filters = [64, 256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=False)
        self.base_size = 512
        self.crop_size = 512
        # self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}

        self.mix = nn.Parameter(torch.FloatTensor(5))
        self.mix.data.fill_(1)

        self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        # self.droupt = nn.Dropout2d(p=0.5)

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4

        # Decoder
        self.decoder0_1 = DecoderBlock(in_channels=filters[1]+filters[0],
                                       out_channels=filters[0],
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)
        self.up1_0 = DUpsampling(256, 256, scale_factor=2)

        self.decoder1_1 = DecoderBlock(in_channels=filters[2]+filters[1],
                                       out_channels=filters[1],
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)
        self.up2_0 = DUpsampling(512, 512, scale_factor=2)

        self.decoder0_2 = DecoderBlock(in_channels=filters[1]+filters[0]+filters[0],
                                       out_channels=filters[0]+filters[0],
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)
        self.up1_1 = DUpsampling(256, 256, scale_factor=2)

        self.decoder2_1 = DecoderBlock(in_channels=filters[3]+filters[2],
                                       out_channels=filters[0]+filters[0],
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)
        self.up3_0 = DUpsampling(1024, 1024, scale_factor=2)

        self.decoder1_2 = DecoderBlock(in_channels=filters[1]+filters[1]+filters[0]+filters[0],
                                       out_channels=filters[0]+filters[0],
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)
        self.up2_1 = DUpsampling(128, 128, scale_factor=2)

        self.decoder0_3 = DecoderBlock(in_channels=filters[1]+filters[0]+filters[0],
                                       out_channels=filters[0]+filters[0],
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)
        self.up1_2 = DUpsampling(128, 128, scale_factor=2)

        # self.decoder3_1 = DecoderBlock(in_channels=filters[4]+filters[3],
        #                                out_channels=filters[1],
        #                                kernel_size=decoder_kernel_size,
        #                                is_deconv=is_deconv)
        # self.decoder2_2 = DecoderBlock(in_channels=filters[2]+filters[1]+filters[0]+filters[0],
        #                                out_channels=filters[2],
        #                                kernel_size=decoder_kernel_size,
        #                                is_deconv=is_deconv)
        # self.decoder1_3 = DecoderBlock(in_channels=filters[2]+filters[1]+filters[1]+filters[0]+filters[0],
        #                                out_channels=filters[2],
        #                                kernel_size=decoder_kernel_size,
        #                                is_deconv=is_deconv)
        # self.decoder0_4 = DecoderBlock(in_channels=filters[2]+filters[1]+filters[0]+filters[0],
        #                                out_channels=filters[2],
        #                                kernel_size=decoder_kernel_size,
        #                                is_deconv=is_deconv)

        self.logit1 = nn.Conv2d(64, num_class, kernel_size=1)
        self.logit2 = nn.Conv2d(128, num_class, kernel_size=1)
        self.logit3 = nn.Conv2d(128, num_class, kernel_size=1)
        # self.logit4 = nn.Conv2d(512, num_class, kernel_size=1)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        _, _, H, W = x.shape
        # stem
        x = self.firstconv(x)  # subsample
        x = self.firstbn(x)
        x_ = self.firstrelu(x)

        # Encoder
        x = self.firstmaxpool(x_)  # 64
        e1 = self.encoder1(x)  # 64
        e2 = self.encoder2(e1)  # 128
        e3 = self.encoder3(e2)  # 256
        # e4 = self.encoder4(e3)  # 512
        # --------Unet Plus Plus Decoder----------------------------------------------

        x0_0 = x_
        # print('x0_0', x0_0.shape)
        x1_0 = e1
        # print('x1_0', x1_0.shape)  # 64 128 128
        x0_1 = self.decoder0_1([x0_0, self.up1_0(x1_0)])  # 256 256
        # print('x0_1', x0_1.shape)

        x2_0 = e2
        # print('x2_0', x2_0.shape)
        x1_1 = self.decoder1_1([x1_0, self.up2_0(x2_0)])
        # print('x1_1', x1_1.shape)
        x0_2 = self.decoder0_2([x0_0, x0_1, self.up1_1(x1_1)])
        # print('x0_2', x0_2.shape)

        x3_0 = e3
        # print('x3_0:', x3_0.shape)
        x2_1 = self.decoder2_1([x2_0,  self.up3_0(x3_0)])
        # print('x2_1:', x2_1.shape)
        x1_2 = self.decoder1_2([x1_0, x1_1, self.up2_1(x2_1)])
        # print('x1_2:', x1_2.shape)
        x0_3 = self.decoder0_3([x0_0, x0_1, x0_2, self.up1_2(x1_2)])
        # print('x0_3:', x0_3.shape)
        #
        # x4_0 = e4
        # # print('x4_0:', x4_0.shape)
        # x3_1 = self.decoder3_1([x3_0, self.up4_0(x4_0)])
        # # print('x3_1:', x3_1.shape)
        # x2_2 = self.decoder2_2([x2_0, x2_1, self.up3_1(x3_1)])
        # # print('x2_2:', x2_2.shape)
        # x1_3 = self.decoder1_3([x1_0, x1_1, x1_2, self.up2_2(x2_2)])
        # # print('x1_3:', x1_3.shape)
        # x0_4 = self.decoder0_4([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)])
        # # print('x0_4:', x0_4.shape)

        logit1 = self.logit1(x0_1)
        logit2 = self.logit2(x0_2)
        logit3 = self.logit3(x0_3)
        # logit4 = self.logit4(x0_4)
        # print(self.mix)
        logit = self.mix[1] * logit1 + self.mix[2] * logit2 + self.mix[3] * logit3
        logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False)

        return logit


# MyResNet34UnetPlus = ResNet34UnetPlus(num_class=2)
# print(MyResNet34UnetPlus)


