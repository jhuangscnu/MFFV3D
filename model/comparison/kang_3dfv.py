import torch
import torch.nn as nn

class Kang_3DFV(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(Kang_3DFV, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        # Layer 1: Initial Convolution
        # Input 224x224x1 -> Output 112x112x32
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2: Max Pooling
        # Output 56x56x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layers 3-4: Depthwise Separable Convolution Block 1
        # Output 56x56x64
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32), # Depthwise
            nn.Conv2d(32, 64, kernel_size=1, stride=1),                      # Pointwise
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Layer 5: Max Pooling
        # Output 28x28x64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layers 6-7: Depthwise Separable Convolution Block 2
        # Output 28x28x128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Layer 8: Max Pooling
        # Output 14x14x128
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layers 9-10: Depthwise Separable Convolution Block 3
        # Output 14x14x256
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Layer 11: Max Pooling
        # Output 7x7x256
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layers 12-13: Depthwise Separable Convolution Block 4
        # Output 7x7x256
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Layer 14: Max Pooling
        # Output 3x3x256 (for 224x224 input)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layers 15-16: Fully Connected Layers
        self.f = nn.Sequential(
            nn.Flatten(),
            # 3*3*256 = 2304 units
            nn.Linear(3 * 3 * 256, 256), 
        )
        self.dp = nn.Dropout(0.5)
        self.f2c = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.block1(x)
        x = self.pool2(x)
        x = self.block2(x)
        x = self.pool3(x)
        x = self.block3(x)
        x = self.pool4(x)
        x = self.block4(x)
        x = self.pool5(x)
        f = self.f(x)
        
        c = self.dp(f)
        c = self.f2c(c)
        return c, f

# x = torch.randn((1, 1, 224, 224))
# model = Kang_3DFV(in_channels=1, num_classes=5000)
# c, f = model(x)
# print(c.shape)
# print(f.shape)