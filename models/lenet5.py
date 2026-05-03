from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        image_size: int = 28,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        feature_dim = self._infer_feature_dim(input_channels, image_size)
        self.fc1 = nn.Linear(feature_dim, 120)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(84, num_classes)

    def _infer_feature_dim(self, input_channels: int, image_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, image_size, image_size)
            features = self._forward_features(dummy)
            return int(features.view(1, -1).size(1))

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        return self.fc3(x)
