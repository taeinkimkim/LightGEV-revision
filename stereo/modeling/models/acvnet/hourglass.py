import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodule import *


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.attention_block = attention_block(channels_3d=in_channels * 4, num_heads=16, block=(4, 4, 4))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def match_volume_shape(self, a, b):
        """
        a, b: [B, C, D, H, W]
        공간 차원(D, H, W)이 다를 경우, 공통 최소 크기로 center crop 해서 맞춘다.
        """
        if a.shape[2:] == b.shape[2:]:
            return a, b

        D = min(a.size(2), b.size(2))
        H = min(a.size(3), b.size(3))
        W = min(a.size(4), b.size(4))

        a = a[:, :, :D, :H, :W]
        b = b[:, :, :D, :H, :W]
        return a, b

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        
        conv4 = self.attention_block(conv4)
        
        # conv5 branch
        a = self.conv5(conv4)
        b = self.redir2(conv2)
        a, b = self.match_volume_shape(a, b)
        conv5 = F.relu(a + b, inplace=True)

        # conv6 branch
        a2 = self.conv6(conv5)
        b2 = self.redir1(x)
        a2, b2 = self.match_volume_shape(a2, b2)
        conv6 = F.relu(a2 + b2, inplace=True)

        return conv6