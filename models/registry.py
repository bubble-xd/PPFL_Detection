from __future__ import annotations

from typing import Dict, List

from .lenet5 import LeNet5
from .resnet import resnet18, resnet34
from .resnet20 import resnet20
from .vgg11 import vgg11


def get_model_names() -> List[str]:
    return ["lenet5", "resnet20", "resnet18", "resnet34", "vgg11"]


def build_model(
    model_name: str,
    input_channels: int,
    num_classes: int,
    image_size: int,
):
    normalized_name = str(model_name).strip().lower()
    if normalized_name == "lenet5":
        return LeNet5(
            input_channels=input_channels,
            num_classes=num_classes,
            image_size=image_size,
        )
    if normalized_name == "resnet20":
        return resnet20(
            input_channels=input_channels,
            num_classes=num_classes,
        )
    if normalized_name == "resnet18":
        return resnet18(
            input_channels=input_channels,
            num_classes=num_classes,
            image_size=image_size,
        )
    if normalized_name == "resnet34":
        return resnet34(
            input_channels=input_channels,
            num_classes=num_classes,
            image_size=image_size,
        )
    if normalized_name == "vgg11":
        return vgg11(
            input_channels=input_channels,
            num_classes=num_classes,
            image_size=image_size,
        )
    raise ValueError(f"Unsupported model: {model_name}")
