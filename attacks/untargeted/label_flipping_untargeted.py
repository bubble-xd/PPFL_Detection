# 参考文献:Data Poisoning Attacks Against Federated Learning Systems
import torch
from torch import Tensor


def label_flipping_untargeted(
    targets: Tensor,
    num_classes: int,
    poison_ratio: float = 1.0,
    inplace: bool = True,
) -> Tensor:
    """
    Label Flipping 攻击：标签翻转（数据投毒）

    将被选中的标签翻转到下一个类别:
        label_new = (label + 1) % num_classes

    参数:
        targets: 原始标签张量
        num_classes: 数据集总类别数
        poison_ratio: 投毒比例 (0.0 - 1.0)，表示多少比例的样本会被翻转标签
        inplace: 是否原地修改目标张量，默认 True

    返回:
        翻转后的标签张量
    """
    if not torch.is_tensor(targets):
        raise TypeError("targets 必须是 Tensor。")
    if not targets.dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("targets 必须是整型标签张量。")
    if num_classes <= 1:
        raise ValueError("num_classes 必须大于 1。")
    if not (0.0 <= poison_ratio <= 1.0):
        raise ValueError("poison_ratio 必须在 [0.0, 1.0] 范围内。")

    poisoned = targets if inplace else targets.clone()
    flat_targets = poisoned.reshape(-1)
    total = flat_targets.numel()
    if total == 0 or poison_ratio == 0.0:
        return poisoned

    poison_count = int(total * poison_ratio)
    if poison_ratio > 0.0 and poison_count == 0:
        poison_count = 1
    poison_count = min(poison_count, total)

    perm = torch.randperm(total, device=flat_targets.device)
    idx = perm[:poison_count]
    flat_targets[idx] = (flat_targets[idx] + 1) % int(num_classes)
    return poisoned


if __name__ == "__main__":
    torch.manual_seed(7)
    y = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    y_flip = label_flipping_untargeted(y, num_classes=10, poison_ratio=1, inplace=False)
    print("before:", y)
    print("after :", y_flip)
