import math
import random

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


class DBAAttack:
    """
    DBA (Distributed Backdoor Attack)

    训练阶段:
    1) 先定义固定预算的全局 trigger
    2) 再把全局 trigger 拆成多个不重叠的局部 trigger shard
    2) 被选中样本统一改成 target_label

    评估阶段:
    - 使用所有局部 shard 的并集评估 all-to-one ASR
    """

    def __init__(
        self,
        dataset_name="mnist",
        target_label=0,
        poisoning_ratio=0.1,
        trigger_size=5,
        shard_id=0,
        num_shards=1,
    ):
        self.dataset_name = dataset_name.lower()
        self.target_label = int(target_label)
        self.poisoning_ratio = float(poisoning_ratio)
        self.trigger_size = int(trigger_size)
        self.num_shards = max(1, int(num_shards))
        self.shard_id = int(shard_id)
        self.attack_model = "targeted"
        self.poison_indices = []

        if not (0.0 <= self.poisoning_ratio <= 1.0):
            raise ValueError("poisoning_ratio 必须在 [0, 1] 范围内。")
        if self.trigger_size <= 0:
            raise ValueError("trigger_size 必须为正整数。")

        self._setup_dataset_meta()
        self._setup_synthesizer()

    def _setup_dataset_meta(self):
        if "mnist" in self.dataset_name:
            self.num_channels = 1
            self.image_size = 28
            self.num_classes = 10
            self.mean = (0.1307,)
            self.std = (0.3081,)
        elif "cifar100" in self.dataset_name:
            # DBA 的触发器分片与数据集类别语义无关；
            # CIFAR100 仅需要使用正确类别数和归一化参数。
            self.num_channels = 3
            self.image_size = 32
            self.num_classes = 100
            self.mean = (0.5071, 0.4867, 0.4408)
            self.std = (0.2675, 0.2565, 0.2761)
        elif "cifar10" in self.dataset_name:
            self.num_channels = 3
            self.image_size = 32
            self.num_classes = 10
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2470, 0.2435, 0.2616)
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")

        if not 0 <= self.target_label < self.num_classes:
            raise ValueError(
                f"target_label 必须落在 [0, {self.num_classes - 1}]，收到 {self.target_label}。"
            )

    @staticmethod
    def _split_length(length, parts):
        base = length // parts
        remainder = length % parts
        bounds = []
        start = 0
        for idx in range(parts):
            chunk = base + (1 if idx < remainder else 0)
            end = start + chunk
            bounds.append((start, end))
            start = end
        return bounds

    def _resolve_tile_bounds(self):
        effective_shards = min(self.num_shards, self.trigger_size * self.trigger_size)
        shard_index = self.shard_id % effective_shards

        grid_cols = math.ceil(math.sqrt(effective_shards))
        grid_rows = math.ceil(effective_shards / grid_cols)

        row_sizes = self._split_length(self.trigger_size, grid_rows)
        col_sizes = self._split_length(self.trigger_size, grid_cols)

        row_idx = shard_index // grid_cols
        col_idx = shard_index % grid_cols
        row_start, row_end = row_sizes[row_idx]
        col_start, col_end = col_sizes[col_idx]
        return row_start, row_end, col_start, col_end

    def _setup_synthesizer(self):
        patch = torch.ones(
            (self.num_channels, self.trigger_size, self.trigger_size)
        )
        norm_transform = transforms.Normalize(self.mean, self.std)
        patch = norm_transform(patch)
        self.trigger_pos = (-self.trigger_size, -self.trigger_size)

        self.trigger = patch.clone()
        self.local_trigger = torch.zeros_like(self.trigger)
        row_start, row_end, col_start, col_end = self._resolve_tile_bounds()
        self.local_trigger[:, row_start:row_end, col_start:col_end] = self.trigger[
            :, row_start:row_end, col_start:col_end
        ]
        self.local_trigger_bounds = (row_start, row_end, col_start, col_end)

    def _write_patch(self, canvas, patch, position):
        row_start, col_start = position
        row_end = row_start + self.trigger_size
        col_end = col_start + self.trigger_size
        canvas[:, row_start:row_end, col_start:col_end] = patch
        return canvas

    def _resolve_trigger_window(self, image):
        c, h, w = image.shape
        if c != self.num_channels:
            raise ValueError(f"图像通道数 {c} 与触发器通道数 {self.num_channels} 不匹配。")
        if self.trigger_size > h or self.trigger_size > w:
            raise ValueError("trigger_size 大于输入图像尺寸。")

        row_start, col_start = self.trigger_pos
        row_start = int(row_start)
        col_start = int(col_start)
        if row_start < 0:
            row_start = h + row_start
        if col_start < 0:
            col_start = w + col_start
        row_start = max(0, min(h - self.trigger_size, row_start))
        col_start = max(0, min(w - self.trigger_size, col_start))
        row_end = row_start + self.trigger_size
        col_end = col_start + self.trigger_size
        return row_start, row_end, col_start, col_end

    def _implant_trigger_tensor(self, image, trigger):
        row_start, row_end, col_start, col_end = self._resolve_trigger_window(image)
        trigger = trigger.to(device=image.device, dtype=image.dtype)
        image[:, row_start:row_end, col_start:col_end] = torch.where(
            trigger != 0,
            trigger,
            image[:, row_start:row_end, col_start:col_end],
        )
        return image

    def _implant_local_trigger(self, image):
        return self._implant_trigger_tensor(image, self.local_trigger)

    def _implant_trigger(self, image):
        return self._implant_trigger_tensor(image, self.trigger)

    def _select_poison_indices(self, labels, train=True):
        if train:
            candidate_indices = list(range(len(labels)))
        else:
            candidate_indices = (
                (labels != self.target_label).nonzero(as_tuple=True)[0].tolist()
            )

        if not candidate_indices:
            return []

        if not train:
            return candidate_indices

        num_poison = int(len(candidate_indices) * self.poisoning_ratio)
        if self.poisoning_ratio > 0 and num_poison == 0:
            num_poison = 1
        num_poison = min(num_poison, len(candidate_indices))
        if num_poison <= 0:
            return []
        return random.sample(candidate_indices, num_poison)

    def poison_batch(self, images, labels, train=True):
        """按 batch 在线执行 DBA 投毒。"""
        poisoned_images = images.clone()
        poisoned_labels = labels.clone()
        active_indices = self._select_poison_indices(poisoned_labels, train=train)
        self.poison_indices = list(active_indices)

        for idx in active_indices:
            if train:
                poisoned_images[idx] = self._implant_local_trigger(poisoned_images[idx])
            else:
                poisoned_images[idx] = self._implant_trigger(poisoned_images[idx])
            poisoned_labels[idx] = self.target_label

        return poisoned_images, poisoned_labels

    def poison_dataset(self, dataset, train=True):
        if isinstance(dataset, TensorDataset):
            images, labels = dataset.tensors
        else:
            loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            images, labels = next(iter(loader))

        eval_indices = self._select_poison_indices(labels, train=False) if not train else None
        poisoned_images, poisoned_labels = self.poison_batch(images, labels, train=train)

        if not train:
            # ASR 评估只统计真正被全局 trigger 改写的非目标类样本，避免目标类干净样本抬高基线。
            if not eval_indices:
                print("Warning: No candidate samples found for DBA ASR evaluation.")
                return TensorDataset(poisoned_images[:0], poisoned_labels[:0])
            eval_index_tensor = torch.as_tensor(eval_indices, dtype=torch.long)
            poisoned_images = poisoned_images[eval_index_tensor].clone()
            poisoned_labels = poisoned_labels[eval_index_tensor].clone()

        if len(poisoned_labels) == 0:
            print("Warning: No candidate samples found for DBA poisoning.")
        return TensorDataset(poisoned_images, poisoned_labels)

    def get_poisoned_loader(self, dataset, batch_size=64, train=True, shuffle=True):
        poisoned_dataset = self.poison_dataset(dataset, train=train)
        return DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=shuffle)
