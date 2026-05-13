from __future__ import annotations

from torch import nn
from torchvision.models import vgg11 as tv_vgg11


def _reset_conv_weights(conv: nn.Conv2d) -> None:
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def vgg11(
    input_channels: int = 3,
    num_classes: int = 10,
    image_size: int = 32,
):
    del image_size

    model = tv_vgg11(weights=None, num_classes=num_classes)

    if input_channels != 3:
        original_conv = model.features[0]
        if not isinstance(original_conv, nn.Conv2d):
            raise TypeError("VGG11 的第一层不是 Conv2d，无法替换输入通道数。")
        model.features[0] = nn.Conv2d(
            input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=True,
        )
        _reset_conv_weights(model.features[0])

    return model
