from typing import Dict, Iterable, Optional

import torch
from torch import Tensor


def additive_noise_attack(
    param_dict: Dict[str, Tensor],
    mean: float = 0.0,
    std: float = 0.1,
    target_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Tensor]:
    """
    Additive Noise 攻击: 向原始参数/更新添加高斯噪声。

    与 random_gradient_attack 的区别:
    - random_gradient_attack: 用噪声替换原值
    - additive_noise_attack: 在原值上叠加噪声
    """
    if not isinstance(param_dict, dict):
        raise TypeError("param_dict 必须是 Dict[str, Tensor]。")
    if std < 0:
        raise ValueError("std 不能为负数。")

    # 噪声攻击会为目标 key 生成新张量，未命中的 key 可复用原引用，避免先全量 clone 再覆盖。
    result = dict(param_dict)
    keys = list(result.keys()) if target_keys is None else target_keys

    for key in keys:
        if key not in result:
            continue
        value = result[key]
        if not torch.is_tensor(value):
            raise TypeError(f"{key} 对应的值不是 Tensor。")
        if not value.dtype.is_floating_point:
            continue

        noise = torch.randn_like(value) * std + mean
        result[key] = value + noise

    return result


if __name__ == "__main__":
    test = {
        "layer.0.weight": torch.ones(4, dtype=torch.float32),
        "layer.1.bias": torch.zeros(2, dtype=torch.float32),
    }
    out = additive_noise_attack(test, mean=0.0, std=0.2)
    print("before:", test)
    print("after :", out)
