import torch
import torch.nn as nn
import torch.nn.functional as F

# Ref: https://github.com/princeton-vl/RAFT/blob/master/core/extractor.py
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes, affine=False)
            self.norm2 = nn.InstanceNorm2d(planes, affine=False)
            self.norm3 = nn.InstanceNorm2d(planes, affine=False)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        """
        ResidualBlock dimension flow:
        Input: (B, in_planes, H, W)
          ↓ conv1 (in_planes→planes, stride=stride)
        (B, planes, H', W')  # H'=H/stride, W'=W/stride
          ↓ norm1 + relu
        (B, planes, H', W')
          ↓ conv2 (planes→planes, stride=1)
        (B, planes, H', W')
          ↓ norm2 + relu
        (B, planes, H', W')
          ↓ downsample (in_planes→planes, stride=stride) + residual connection
        Output: (B, planes, H', W')
        """
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        x = self.downsample(x)

        return self.relu(x+y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64, affine=False)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=1)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        ResNet-14, like ResNet-18/34, but with 3 layers instead of 4.
        Dimension flow:
        Input: (B, 3, H, W)
          ↓ conv1 (3→64, stride=2)
        (B, 64, H/2, W/2)
          ↓ norm1 + relu1
        (B, 64, H/2, W/2)
          ↓ layer1 (64→64, stride=1)
        (B, 64, H/2, W/2)
          ↓ layer2 (64→96, stride=2)
        (B, 96, H/4, W/4)
          ↓ layer3 (96→128, stride=1)
        (B, 128, H/4, W/4)
          ↓ conv2 (128→output_dim, 1x1)
        Output: (B, output_dim, H/4, W/4)
        """
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # (B, 3, H, W) -> (B, 64, H/2, W/2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # (B, 64, H/2, W/2) -> (B, 64, H/2, W/2)
        x = self.layer1(x)
        # (B, 64, H/2, W/2) -> (B, 96, H/4, W/4)
        x = self.layer2(x)
        # (B, 96, H/4, W/4) -> (B, 128, H/4, W/4)
        x = self.layer3(x)

        # (B, 128, H/4, W/4) -> (B, output_dim, H/4, W/4)
        x = self.conv2(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, x.shape[0]//2, dim=0)

        return x