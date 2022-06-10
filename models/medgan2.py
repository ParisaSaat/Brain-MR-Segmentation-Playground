import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import torchvision.models as models

"""
Code from https://github.com/ExplainableML/UncerGuidedI2I/blob/cc4a4e2d0b84e97b072534b46337d7c7a50d18e0/src/networks.py#L118
"""


class ResConv(nn.Module):
    """
    Residual convolutional block, where
    convolutional block consists: (convolution => [BN] => ReLU) * 3
    residual connection adds the input to the output
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x_in = self.double_conv1(x)
        x1 = self.double_conv(x)
        return self.double_conv(x) + x_in

class Down(nn.Module):
    """Downscaling with maxpool then Resconv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
	"""Upscaling then double conv"""
	def __init__(self, in_channels, out_channels, bilinear=True):
		super().__init__()
		# if bilinear, use the normal convolutions to reduce the number of channels
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			self.conv = ResConv(in_channels, out_channels, in_channels // 2)
		else:
			self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
			self.conv = ResConv(in_channels, out_channels)
	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]
		x1 = F.pad(
			x1, 
			[
				diffX // 2, diffX - diffX // 2,
				diffY // 2, diffY - diffY // 2
			]
		)
		# if you have padding issues, see
		# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
		# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
	def forward(self, x):
		# return F.relu(self.conv(x))
		return self.conv(x)

##### The composite networks
class UNet(nn.Module):
	def __init__(self, n_channels, out_channels, bilinear=True):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.out_channels = out_channels
		self.bilinear = bilinear
		####
		self.inc = ResConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		factor = 2 if bilinear else 1
		self.down4 = Down(512, 1024 // factor)
		self.up1 = Up(1024, 512 // factor, bilinear)
		self.up2 = Up(512, 256 // factor, bilinear)
		self.up3 = Up(256, 128 // factor, bilinear)
		self.up4 = Up(128, 64, bilinear)
		self.outc = OutConv(64, out_channels)
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
		y = self.outc(x)
		return y

class CasUNet(nn.Module):
	def __init__(self, n_unet, io_channels, bilinear=True):
		super(CasUNet, self).__init__()
		self.n_unet = n_unet
		self.io_channels = io_channels
		self.bilinear = bilinear
		####
		self.unet_list = nn.ModuleList()
		for i in range(self.n_unet):
			self.unet_list.append(UNet(self.io_channels, self.io_channels, self.bilinear))
	def forward(self, x):
		y = x
		for i in range(self.n_unet):
			if i==0:
				y = self.unet_list[i](y)
			else:
				y = self.unet_list[i](y+x)
		return y

class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc, ndf=64, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the first conv layer
        """
        super().__init__()
        
        use_bias = True
        kw = 4
        padw = 1
        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw) 
        self.lrelu1 = nn.LeakyReLU(0.2, True)
        self.inst_norm1 = norm_layer(ndf)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.lrelu2 = nn.LeakyReLU(0.2, True)
        self.inst_norm2 = norm_layer(ndf*2)

        self.conv3 = nn.Conv2d(ndf * 2, 1, kernel_size=kw, stride=4, padding=padw)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        """Standard forward."""
        features = []
        
        x = self.conv1(input)
        x = self.lrelu1(x)
        x = self.inst_norm1(x)
        features.append(x)

        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.inst_norm2(x)
        features.append(x)

        x = self.conv3(x)
        
        return self.sigmoid(x), features


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        # Store relevant features
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features

# import numpy as np
# model = CasUNet(n_unet = 2, io_channels = 1)
# input = torch.zeros((1, 1, 256, 256))
# output = model(input)
# print(output.shape)
# # print()