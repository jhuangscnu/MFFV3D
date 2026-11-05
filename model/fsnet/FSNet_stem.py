import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
# from numpy.lib.arraypad import pad
from numpy import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential



class Stem(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super(Stem, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(int(out_channels//2))
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        
        return out
        
class FFBlock(nn.Module):
    def __init__(self, dim, size=None, out_channels=None, norm_layer=None):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.s = size
        self.act = nn.LeakyReLU(inplace=True)
        self.complex_weight = nn.Parameter(torch.randn(self.s, int((self.s/2)+1), out_channels, 2, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.transpose(1, 3)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight) # learnable global filters
        x = x * weight
        x = torch.fft.irfft2(x, s=(self.s, self.s), dim=(1, 2), norm='ortho')
        x = x.transpose(3, 1)
        # x = self.norm(x)
        # x = self.act(x)

        return x

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, expansion=None, groups=None, norm_layer=None):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=expansion*in_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=expansion*in_channels, out_channels=expansion*in_channels, kernel_size=3, groups=groups, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=expansion*in_channels, out_channels=out_channels, kernel_size=1, bias=False)

        self.bn1 = norm_layer(expansion*in_channels)
        self.bn2 = norm_layer(expansion*in_channels)
        self.bn3 = norm_layer(out_channels)

        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)

        return x

class DSBlock(nn.Module):
    def __init__(self, size=None, in_channels=None, out_channels=None, expansion=None, bot_groups=None, norm_layer=None, fuse_type='DAF', f_ablation=None, s_ablation=None, fa_ablation=None, sa_ablation=None, w_mode=None):
        super(DSBlock, self).__init__()
        self.size = size
        self.f_ablation = f_ablation
        self.s_ablation = s_ablation
        self.fa_ablation = fa_ablation
        self.sa_ablation = sa_ablation

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # fusion
        if fuse_type == 'AFF':
            self.fuse_mode = AFF(channels=out_channels, w_mode=w_mode)
        elif fuse_type == 'aAFF':
            self.fuse_mode = aAFF(channels=out_channels)
        elif fuse_type == 'iAFF':
            self.fuse_mode = iAFF(channels=out_channels)
        elif fuse_type == 'DAF':
            self.fuse_mode = DAF()
        else:
            self.fuse_mode = None

        if self.f_ablation or self.fa_ablation:
            self.bot = BottleneckBlock(in_channels=in_channels, out_channels=out_channels, expansion=expansion, groups=bot_groups, norm_layer=norm_layer)
        elif self.s_ablation or self.sa_ablation:
            self.ff = FFBlock(dim=out_channels, size=self.size, out_channels=out_channels)
            self.pre_ff = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        else:
            self.bot = BottleneckBlock(in_channels=in_channels, out_channels=out_channels, expansion=expansion, groups=bot_groups, norm_layer=norm_layer)
            self.ff = FFBlock(dim=out_channels, size=self.size, out_channels=out_channels)
            self.pre_ff = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        if self.f_ablation:
            out = self.bot(x)
            out = self.fuse_mode(out, out)
            # print('fsnet-f')
        elif self.s_ablation:
            out = self.ff(self.pre_ff(x))
            out = self.fuse_mode(out, out)
            # print('fsnet-s')
        elif self.fa_ablation:
            out = self.bot(x)
            # print('fsnet-fa')
        elif self.sa_ablation:
            out = self.ff(self.pre_ff(x))
            # print('fsnet-sa')
        else:
            bot_out = self.bot(x)
            ff_out = self.ff(self.pre_ff(x))
            out = self.fuse_mode(bot_out, ff_out)
            # print('fsnet')
        return out



class DAF(nn.Module):
    '''
     DirectAddFuse
    '''

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual



class AFF(nn.Module):
    '''
    AFF
    '''

    def __init__(self, channels=64, r=2, w_mode=None):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.w_mode = w_mode

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels), # 有这句会报错，因为BN不允许只有一个数据点的操作
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels), # 有这句会报错，因为BN不允许只有一个数据点的操作
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        
        if self.w_mode == 'x':
            xo = x * wei + residual * (1 - wei)
        elif self.w_mode == 'r':
            xo = x * (1 - wei) + residual * wei
        else:
            return None
        return xo

class aAFF(nn.Module):
    '''
    自适应多特征融合 aAFF
    '''

    def __init__(self, channels=64, r=2):
        super(aAFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels), # 有这句会报错，因为BN不允许只有一个数据点的操作
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels), # 有这句会报错，因为BN不允许只有一个数据点的操作
        )

        self.sigmoid = nn.Sigmoid()

        self.weight = nn.Parameter(torch.randn(1, dtype=torch.float32), requires_grad=True)

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = self.weight * x * wei + (1-self.weight) * residual * (1 - wei)
        return xo

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=2):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),# 有这句会报错，因为BN不允许只有一个数据点的操作
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),# 有这句会报错，因为BN不允许只有一个数据点的操作
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),# 有这句会报错，因为BN不允许只有一个数据点的操作
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),# 有这句会报错，因为BN不允许只有一个数据点的操作
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

class FSNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1000, norm_layer=None, a_ablation=False, f_ablation=None, s_ablation=None, fa_ablation=None, sa_ablation=None, w_mode=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stem = Stem(in_channels=in_channels, out_channels=32)

        if a_ablation:
            self.ds1 = DSBlock(size=56, in_channels=32, out_channels=64, expansion=2, bot_groups=4, fuse_type='DAF', f_ablation=f_ablation, s_ablation=s_ablation, fa_ablation=fa_ablation, sa_ablation=sa_ablation)
            self.ds2 = DSBlock(size=28, in_channels=64, out_channels=128, expansion=2, bot_groups=8, fuse_type='DAF', f_ablation=f_ablation, s_ablation=s_ablation, fa_ablation=fa_ablation, sa_ablation=sa_ablation)
            self.ds3 = DSBlock(size=14, in_channels=128, out_channels=256, expansion=2, bot_groups=16, fuse_type='DAF', f_ablation=f_ablation, s_ablation=s_ablation, fa_ablation=fa_ablation, sa_ablation=sa_ablation)
            # self.ds4 = DSBlock(size=7, in_channels=128, out_channels=256, expansion=2, bot_groups=32, fuse_type='DAF', f_ablation=f_ablation, s_ablation=s_ablation, fa_ablation=fa_ablation, sa_ablation=sa_ablation)
            # print('fsnet-a')
        elif fa_ablation or sa_ablation:
            self.ds1 = DSBlock(size=56, in_channels=32, out_channels=64, expansion=2, bot_groups=4, fuse_type=None, f_ablation=f_ablation, s_ablation=s_ablation, fa_ablation=fa_ablation, sa_ablation=sa_ablation, w_mode=w_mode)
            self.ds2 = DSBlock(size=28, in_channels=64, out_channels=128, expansion=2, bot_groups=8, fuse_type=None, f_ablation=f_ablation, s_ablation=s_ablation, fa_ablation=fa_ablation, sa_ablation=sa_ablation, w_mode=w_mode)
            self.ds3 = DSBlock(size=14, in_channels=128, out_channels=256, expansion=2, bot_groups=16, fuse_type=None, f_ablation=f_ablation, s_ablation=s_ablation, fa_ablation=fa_ablation, sa_ablation=sa_ablation, w_mode=w_mode)
            # self.ds4 = DSBlock(size=7, in_channels=128, out_channels=256, expansion=2, bot_groups=32, fuse_type=None, f_ablation=f_ablation, s_ablation=s_ablation, fa_ablation=fa_ablation, sa_ablation=sa_ablation)
        else:
            self.ds1 = DSBlock(size=56, in_channels=32, out_channels=64, expansion=2, bot_groups=4, fuse_type='AFF', f_ablation=f_ablation, s_ablation=s_ablation, fa_ablation=fa_ablation, sa_ablation=sa_ablation, w_mode=w_mode)
            self.ds2 = DSBlock(size=28, in_channels=64, out_channels=128, expansion=2, bot_groups=8, fuse_type='AFF', f_ablation=f_ablation, s_ablation=s_ablation, fa_ablation=fa_ablation, sa_ablation=sa_ablation, w_mode=w_mode)
            self.ds3 = DSBlock(size=14, in_channels=128, out_channels=256, expansion=2, bot_groups=16, fuse_type='AFF', f_ablation=f_ablation, s_ablation=s_ablation, fa_ablation=fa_ablation, sa_ablation=sa_ablation, w_mode=w_mode)
            # self.ds4 = DSBlock(size=7, in_channels=128, out_channels=256, expansion=2, bot_groups=32, fuse_type='AFF', f_ablation=f_ablation, s_ablation=s_ablation, fa_ablation=fa_ablation, sa_ablation=sa_ablation)
            # print('fsnet')

        self.nl1 = norm_layer(64)
        self.nl2 = norm_layer(128)
        self.nl3 = norm_layer(256)
        # self.nl4 = norm_layer(256)

        self.act = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.x2f = nn.AdaptiveAvgPool2d(1)
        self.dp = nn.Dropout(0.5)
        self.f2c = nn.Linear(256, self.num_classes)


    def forward(self, x):
        x = self.stem(x)
        
        x = self.ds1(x)
        x = self.maxpool(x)
        x = self.nl1(x)
        x = self.act(x)

        x = self.ds2(x)
        x = self.maxpool(x)
        x = self.nl2(x)
        x = self.act(x)

        x = self.ds3(x)
        x = self.maxpool(x)
        x = self.nl3(x)
        x = self.act(x)

        # x = self.ds4(x)
        # x = self.maxpool(x)
        # x = self.nl4(x)
        # x = self.act(x)

        f = self.x2f(x)
        f = torch.flatten(f, 1)

        c = self.dp(f)
        c = self.f2c(c)
        # c = F.log_softmax(c, dim=-1)

        return c, f


# #
# x = torch.randn((1, 1, 224, 224))
# model = FSNet(in_channels=1, num_classes=5000, a_ablation=False, f_ablation=False, s_ablation=False, fa_ablation=False, sa_ablation=False, w_mode='x')
# c, f = model(x)
# print(c.shape)
# print(f.shape)
