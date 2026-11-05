import timm
import torch
import torch.nn as nn


class Xception(nn.Module):
    def __init__(self, in_channels=None, num_classes=5000):
        super().__init__()
        self.model = timm.create_model('xception', in_chans=in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dp = nn.Dropout(0.5)
        self.f2c = nn.Linear(2048, num_classes)
    def forward(self, x):
        f = self.model.forward_features(x)
        f = self.pool(f).view(f.size(0), -1)
        # print('shape of f:', f.shape)

        c = self.dp(f)
        c = self.f2c(c)
        return c, f

# x = torch.randn((2, 1, 224, 224))
# model = Xception(in_channels=1, num_classes=5000)
# c, f = model(x)
# print(c.shape)
# print(f.shape)
