import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self, in_ch=3, n_classes=2):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 512)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(512, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(512+32, 32)
        self.conv10 = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        # print('x', x.shape)  # 3 256 256
        c1 = self.conv1(x)
        # print('c1', c1.shape)  # 512 256 256
        p1 = self.pool1(c1)
        # print('p1', p1.shape)  # 512 128 128
        c2 = self.conv2(p1)
        # print('c2', c2.shape)  # 64 128 128
        p2 = self.pool2(c2)
        # print('p2', p2.shape)  # 64 64 64
        c3 = self.conv3(p2)
        # print('c3', c3.shape)  # 128 64 64
        p3 = self.pool3(c3)
        # print('p3', p3.shape)  # 128 32 32
        c4 = self.conv4(p3)
        # print('c4', c4.shape)  # 256 32 32
        p4 = self.pool4(c4)
        # print('p4', p4.shape)  # 256 16 16
        c5 = self.conv5(p4)
        # print('c5', c5.shape)  # 256 16 16

        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        # print('c6', c6.shape)  # 256 16 16
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        # print('c7', c7.shape)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        # print('c8', c8.shape)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        # print('c9', c9.shape)
        c10 = self.conv10(c9)
        out = c10
        return out



