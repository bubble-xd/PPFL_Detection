from typing import Dict, Iterable, Optional

import torch
from torch import Tensor


def scaling_attack(
    update_dict: Dict[str, Tensor],
    gamma: float = 10.0,
    target_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Tensor]:
    """
    梯度放大攻击: delta -> gamma * delta
    统一输入格式为 Dict[str, Tensor]
    """
    if not isinstance(update_dict, dict):
        raise TypeError("update_dict 必须是 Dict[str, Tensor]。")
    if gamma < 0:
        raise ValueError("gamma 不能为负数。")

    # 只替换被攻击的张量，不预先复制整份 update，减少 ResNet/VGG 场景的内存带宽开销。
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

        result[key] = gamma * value

    return result


if __name__ == "__main__":
    fake = {"layer.0.weight": torch.tensor([1.0, -2.0]), "layer.0.bias": torch.tensor([0.5])}
    out = scaling_attack(fake, gamma=20.0)
    print(out)
