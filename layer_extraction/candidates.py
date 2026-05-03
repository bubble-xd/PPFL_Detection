from __future__ import annotations

from typing import List

from torch import nn


def _is_residual_projection_layer(module_name: str) -> bool:
    # shortcut/downsample 是残差分支上的投影层，不计入论文/模型名里的主干层数。
    return ".downsample." in module_name or ".shortcut." in module_name


def get_candidate_layer_prefixes(model: nn.Module) -> List[str]:
    candidate_layers: List[str] = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        # 这里把“层”的粒度固定成带参数的叶子 Conv/Linear，
        # 这样既能和现有 KEY_LAYER_MAP 的使用方式对齐，也能避开 BN 统计量噪声。
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if any(True for _ in module.children()):
            continue
        if not any(True for _ in module.parameters(recurse=False)):
            continue
        if _is_residual_projection_layer(module_name):
            continue
        candidate_layers.append(module_name)

    if not candidate_layers:
        raise ValueError("当前模型未找到可用于 layer extraction 的 Conv2d/Linear 叶子模块。")
    return candidate_layers
