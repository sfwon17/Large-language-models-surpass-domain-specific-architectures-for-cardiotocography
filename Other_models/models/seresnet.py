# cleaned version 
import torch
import torch.nn as nn

class SEResNet152d_1D(nn.Module):
    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 256, blocks=3, stride=1)
        self.layer2 = self._make_layer(256, 512, blocks=8, stride=2)
        self.layer3 = self._make_layer(512, 1024, blocks=36, stride=2)
        self.layer4 = self._make_layer(1024, 2048, blocks=3, stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(
            SEBottleneck1D(in_channels, out_channels, stride=stride, downsample=True)
        )
        for _ in range(1, blocks):
            layers.append(
                SEBottleneck1D(out_channels, out_channels, stride=1, downsample=False)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        logits = self.fc(x)

        return logits


class SEBottleneck1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, reduction=16):
        super().__init__()

        bottleneck_channels = out_channels // 4

        self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(bottleneck_channels)

        self.conv2 = nn.Conv1d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)

        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.se = SEModule1D(out_channels, reduction=reduction)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEModule1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        reduced_channels = max(channels // reduction, 1)

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, _ = x.size()

        y = self.squeeze(x).view(batch, channels)
        y = self.excitation(y).view(batch, channels, 1)

        return x * y.expand_as(x)


def create_seresnet152d_model(num_classes=2, dropout=0.1):
    return SEResNet152d_1D(num_classes=num_classes, dropout=dropout)
