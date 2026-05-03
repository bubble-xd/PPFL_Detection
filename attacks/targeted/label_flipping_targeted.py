# 参考文献:Data Poisoning Attacks Against Federated Learning Systems
import torch
from torch import Tensor


def label_flipping_targeted(
    targets: Tensor,
    target_class: int,
    source_class: int,
    poison_ratio: float = 1.0,
    inplace: bool = True,
) -> Tensor:
    """
    Label Flipping source-to-target 攻击：将选中的源类别样本标签翻转为目标类别。

    参数:
        targets: 原始标签张量
        source_class: 源类别（仅该类别样本会被翻转）
        target_class: 攻击目标类别（要伪装成的类别）
        poison_ratio: 投毒比例 (0.0 - 1.0)，表示全部源类别样本中多少比例会被翻转标签
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
    if not (0.0 <= poison_ratio <= 1.0):
        raise ValueError("poison_ratio 必须在 [0.0, 1.0] 范围内。")
    if int(source_class) == int(target_class):
        raise ValueError("source_class 和 target_class 不能相同。")

    poisoned = targets if inplace else targets.clone()
    flat_targets = poisoned.reshape(-1)

    # 仅将源类别样本作为候选投毒对象
    candidate_indices = (flat_targets == source_class).nonzero(as_tuple=True)[0]
    num_candidates = candidate_indices.numel()

    if num_candidates == 0 or poison_ratio == 0.0:
        return poisoned

    poison_count = int(num_candidates * poison_ratio)
    if poison_ratio > 0.0 and poison_count == 0:
        poison_count = 1
    poison_count = min(poison_count, num_candidates)

    # 随机选择要翻转的索引
    perm = torch.randperm(num_candidates, device=flat_targets.device)
    idx_to_flip = candidate_indices[perm[:poison_count]]
    
    # 执行翻转
    flat_targets[idx_to_flip] = target_class
    
    return poisoned


if __name__ == "__main__":
    torch.manual_seed(7)
    y = torch.tensor([0, 1, 2, 2, 2, 2, 3, 4, 5], dtype=torch.long)
    print("原始标签:", y)

    # 将全部 2 类样本按比例翻转为 9
    y_flip = label_flipping_targeted(
        y,
        target_class=9,
        source_class=2,
        poison_ratio=1,
        inplace=False
    )
    print("翻转后标签:", y_flip)

    print("原始源类别(2)数量:", (y == 2).sum().item())
    print("翻转后目标类别(9)数量:", (y_flip == 9).sum().item())
