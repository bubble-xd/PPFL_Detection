# 使用Dict[str, Tensor]格式的更新输入
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor


def sign_flipping(
    update_dict: Dict[str, Tensor],
    alpha: float = 1.0,
    target_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Tensor]:
    """
    符号反转攻击: delta -> -alpha * delta
    统一输入格式为 Dict[str, Tensor]
    """
    if not isinstance(update_dict, dict):
        raise TypeError("update_dict 必须是 Dict[str, Tensor]。")
    if alpha < 0:
        raise ValueError("alpha 不能为负数。")

    # update 攻击不会原地修改输入；先浅拷贝映射，再只替换目标 key，避免大模型上无意义地 clone 全量 update。
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

        result[key] = -alpha * value

    return result


if __name__ == "__main__":
    test = {"layer.0.weight": torch.tensor([1.0, -2.0]), "layer.0.bias": torch.tensor([0.5])}
    out = sign_flipping(test, alpha=1.0)
    print(out)
