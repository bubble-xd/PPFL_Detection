from __future__ import annotations

from torch import nn
from torchvision.models import resnet18 as tv_resnet18
from torchvision.models import resnet34 as tv_resnet34


def _reset_conv_weights(conv: nn.Conv2d) -> None:
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def _build_resnet(
    model_builder,
    input_channels: int,
    num_classes: int,
    image_size: int,
):
    model = model_builder(weights=None, num_classes=num_classes)

    if int(image_size) <= 64:
        model.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        _reset_conv_weights(model.conv1)
        model.maxpool = nn.Identity()
        return model

    if input_channels != 3:
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(
            input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        _reset_conv_weights(model.conv1)

    return model


def resnet18(
    input_channels: int = 3,
    num_classes: int = 10,
    image_size: int = 32,
):
    return _build_resnet(
        model_builder=tv_resnet18,
        input_channels=input_channels,
        num_classes=num_classes,
        image_size=image_size,
    )


def resnet34(
    input_channels: int = 3,
    num_classes: int = 10,
    image_size: int = 32,
):
    return _build_resnet(
        model_builder=tv_resnet34,
        input_channels=input_channels,
        num_classes=num_classes,
        image_size=image_size,
    )
