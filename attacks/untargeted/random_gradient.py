from typing import Dict, Iterable, Optional

import torch
from torch import Tensor


def random_gradient_attack(
    update_dict: Dict[str, Tensor],
    mean: float = 0.0,
    std: float = 1.0,
    target_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Tensor]:
    """
    高斯随机梯度攻击: 用 N(mean, std^2) 噪声替换上传更新。
    统一输入格式为 Dict[str, Tensor]
    """
    if not isinstance(update_dict, dict):
        raise TypeError("update_dict 必须是 Dict[str, Tensor]。")
    if std < 0:
        raise ValueError("std 不能为负数。")

    # 目标 key 会被随机张量完整替换，浅拷贝映射即可避免一次全量 update clone。
    result = dict(update_dict)
    keys = list(result.keys()) if target_keys is None else target_keys

    for key in keys:
        if key not in result:
            continue
        value = result[key]
        if not torch.is_tensor(value):
            raise TypeError(f"{key} 对应的更新不是 Tensor。")
        if not value.dtype.is_floating_point:
            continue

        result[key] = torch.randn_like(value) * std + mean

    return result


if __name__ == "__main__":
    fake = {"layer.0.weight": torch.ones(4), "layer.1.bias": torch.zeros(2)}
    out = random_gradient_attack(fake, mean=0.0, std=3.0)
    print(out)
