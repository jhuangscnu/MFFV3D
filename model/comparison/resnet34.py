from timm.models import ResNet, BasicBlock
import torch
import torch.nn as nn


class ResNet34_FV(nn.Module):
    def __init__(self, in_channels=None, num_classes=5000):
        super().__init__()
        self.resnet34 = ResNet(block=BasicBlock, in_chans=in_channels, num_classes=num_classes,layers=[3, 4, 6, 3])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dp = nn.Dropout(0.5)
        self.f2c = nn.Linear(512, num_classes)
    def forward(self, x):
        f = self.resnet34.forward_features(x)
        f = self.pool(f).view(f.size(0), -1)

        c = self.dp(f)
        c = self.f2c(c)
        return c, f


# x = torch.randn((2, 2, 224, 224))
# model = ResNet34_FV(in_channels=2)
# c, f = model(x)
# print(c.shape)
# print(f.shape)
