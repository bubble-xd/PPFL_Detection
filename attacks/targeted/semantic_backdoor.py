# 参考文献: How To Backdoor Federated Learning
# https://arxiv.org/abs/1807.00459

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SemanticBackdoorAttack:
    """
    语义后门攻击：
    - 不修改样本语义，仅将带有特定语义特征的样本注入训练集并标为目标标签。
    - 仅支持以下组合：
      1) CIFAR10 + southwest
      2) MNIST + ardis
      3) CIFAR100 + southwest
    """

    def __init__(
        self,
        dataset_name="cifar10",
        target_label=2,
        poisoning_ratio=0.1,
        semantic_source="southwest",
        epsilon=0.25,
        projection_type="l_2",
        scaling_factor=1.0,
    ):
        self.dataset_name = str(dataset_name).strip().lower().replace("-", "")
        self.target_label = int(target_label)
        self.poisoning_ratio = float(poisoning_ratio)
        self.semantic_source = str(semantic_source).strip().lower()
        # 这些参数主要给联邦阶段的 update poisoning 使用，统一挂在攻击对象上便于 adapter 读取。
        self.epsilon = float(epsilon) if epsilon is not None else None
        self.projection_type = str(projection_type).strip().lower()
        self.scaling_factor = float(scaling_factor)

        if not (0.0 <= self.poisoning_ratio <= 1.0):
            raise ValueError("poisoning_ratio 必须在 [0, 1] 范围内。")
        if self.scaling_factor <= 0:
            raise ValueError("scaling_factor 必须为正数。")

        self.train_semantic_samples, self.eval_semantic_samples = self._load_semantic_samples()
        # 保留旧属性名给外部诊断代码读取；训练投毒默认使用训练语义样本池。
        self.semantic_samples = self.train_semantic_samples

    def _load_semantic_samples(self):
        """按配置加载语义样本，并拆分训练注入池和 ASR 评估池。"""
        if self.semantic_source == "southwest":
            if self.dataset_name not in {"cifar10", "cifar100"}:
                raise ValueError("semantic_source='southwest' 仅支持 CIFAR10/CIFAR100。")
            try:
                from attacks.targeted.edge_case import (
                    CIFAR10_MEAN,
                    CIFAR10_STD,
                    CIFAR100_MEAN,
                    CIFAR100_STD,
                    SouthwestAirlineDataset,
                )
            except Exception as exc:
                raise ImportError("无法导入 SouthwestAirlineDataset。") from exc

            # CIFAR100 复用同一组 RGB 语义样本，但必须按 CIFAR100 的统计量标准化。
            if self.dataset_name == "cifar100":
                mean, std = CIFAR100_MEAN, CIFAR100_STD
            else:
                mean, std = CIFAR10_MEAN, CIFAR10_STD
            dataset = SouthwestAirlineDataset(
                target_label=self.target_label,
                mean=mean,
                std=std,
            )
            train_x, _ = dataset.get_poisoned_trainset()
            test_x, _ = dataset.get_poisoned_testset().tensors
            return train_x, test_x

        if self.semantic_source == "ardis":
            if "mnist" not in self.dataset_name:
                raise ValueError("semantic_source='ardis' 仅支持 MNIST。")
            try:
                from attacks.targeted.edge_case import ARDISDataset
            except Exception as exc:
                raise ImportError("无法导入 ARDISDataset。") from exc

            dataset = ARDISDataset(target_label=self.target_label)
            train_x, _ = dataset.get_poisoned_trainset()
            test_x, _ = dataset.get_poisoned_testset().tensors
            return train_x, test_x

        raise ValueError(
            f"不支持的 semantic_source: {self.semantic_source}。"
            "当前仅支持: southwest, ardis。"
        )

    @staticmethod
    def _align_to_reference(semantic_x: torch.Tensor, ref_x: torch.Tensor) -> torch.Tensor:
        """将语义样本对齐到参考数据的通道数和空间尺寸。"""
        if semantic_x.dim() != 4 or ref_x.dim() != 4:
            raise ValueError("输入必须为 4 维张量 (N, C, H, W)。")

        if semantic_x.shape[-2:] != ref_x.shape[-2:]:
            semantic_x = F.interpolate(
                semantic_x,
                size=ref_x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if semantic_x.size(1) != ref_x.size(1):
            if semantic_x.size(1) == 1 and ref_x.size(1) == 3:
                semantic_x = semantic_x.repeat(1, 3, 1, 1)
            elif semantic_x.size(1) == 3 and ref_x.size(1) == 1:
                semantic_x = semantic_x.mean(dim=1, keepdim=True)
            else:
                raise ValueError("语义样本通道数与目标数据不匹配。")

        return semantic_x.to(dtype=ref_x.dtype)

    def poison_dataset(self, dataset, train=True):
        """构造语义后门数据集。"""
        if isinstance(dataset, TensorDataset):
            images, labels = dataset.tensors
        else:
            loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            images, labels = next(iter(loader))

        images = images.clone()
        labels = labels.clone()

        if not train:
            # ASR 使用 held-out 语义样本，避免把训练注入过的样本计入评估分母。
            poison_x = self._align_to_reference(self.eval_semantic_samples, images)
            poison_y = torch.full((len(poison_x),), self.target_label, dtype=torch.long)
            return TensorDataset(poison_x, poison_y)

        num_clean = len(images)
        num_poison = int(num_clean * self.poisoning_ratio)
        if self.poisoning_ratio > 0 and num_poison == 0:
            num_poison = 1
        if num_poison == 0:
            return TensorDataset(images, labels)

        train_samples = self.train_semantic_samples
        if len(train_samples) < num_poison:
            sample_indices = torch.randint(0, len(train_samples), (num_poison,))
        else:
            sample_indices = torch.randperm(len(train_samples))[:num_poison]

        semantic_x = train_samples[sample_indices]
        semantic_x = self._align_to_reference(semantic_x, images)
        semantic_y = torch.full((num_poison,), self.target_label, dtype=torch.long)

        mixed_images = torch.cat([images, semantic_x], dim=0)
        mixed_labels = torch.cat([labels, semantic_y], dim=0)

        return TensorDataset(mixed_images, mixed_labels)

    def poison_batch(self, images: torch.Tensor, labels: torch.Tensor):
        """按 batch 在线注入语义样本，保留 clean batch 并追加 poison 样本。"""
        poisoned_images = images.clone()
        poisoned_labels = labels.clone()

        batch_size = len(poisoned_images)
        num_poison = int(batch_size * self.poisoning_ratio)
        if self.poisoning_ratio > 0 and num_poison == 0:
            num_poison = 1
        if num_poison <= 0:
            return poisoned_images, poisoned_labels

        train_samples = self.train_semantic_samples
        if len(train_samples) < num_poison:
            sample_indices = torch.randint(0, len(train_samples), (num_poison,))
        else:
            sample_indices = torch.randperm(len(train_samples))[:num_poison]

        semantic_x = train_samples[sample_indices]
        semantic_x = self._align_to_reference(semantic_x, poisoned_images)
        semantic_y = torch.full(
            (num_poison,),
            self.target_label,
            dtype=poisoned_labels.dtype,
            device=poisoned_labels.device,
        )

        semantic_x = semantic_x.to(
            device=poisoned_images.device,
            dtype=poisoned_images.dtype,
        )
        mixed_images = torch.cat([poisoned_images, semantic_x], dim=0)
        mixed_labels = torch.cat([poisoned_labels, semantic_y], dim=0)
        return mixed_images, mixed_labels

    def get_poisoned_loader(self, dataset, batch_size=64, train=True, shuffle=True):
        poisoned_dataset = self.poison_dataset(dataset, train=train)
        return DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=shuffle)
