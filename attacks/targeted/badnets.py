import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

class BadNetsAttack:
    """
    BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain
    基于像素图案的后门攻击实现:
    在训练数据集中随机选择一部分样本
    在这些样本的特定位置添加触发器（通常是明显的小方块）
    修改这些样本的标签为目标类别（target label）
    保持其他样本不变
    """
    def __init__(
        self,
        dataset_name="mnist",
        target_label=0,
        poisoning_ratio=0.1,
        trigger_size=5,
    ):
        self.dataset_name = dataset_name.lower()
        self.target_label = int(target_label)
        self.poisoning_ratio = float(poisoning_ratio)
        self.trigger_size = int(trigger_size)
        self.attack_model = "targeted"

        if not (0.0 <= self.poisoning_ratio <= 1.0):
            raise ValueError("poisoning_ratio 必须在 [0, 1] 范围内。")
        if self.trigger_size <= 0:
            raise ValueError("trigger_size 必须为正整数。")

        # 数据集元数据配置
        self._setup_dataset_meta()

        # 初始化触发器和合成器
        self._setup_synthesizer()

    def _setup_dataset_meta(self):
        if "mnist" in self.dataset_name:
            self.num_channels = 1
            self.image_size = 28
            self.num_classes = 10
            self.mean = (0.1307,)
            self.std = (0.3081,)
        elif "cifar100" in self.dataset_name:
            # CIFAR100 仍是 32x32 RGB，触发器逻辑可复用 CIFAR10；
            # 这里只切换类别数和归一化统计量，保证 ASR 输入与训练数据一致。
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

    def _setup_synthesizer(self):
        # 1. 定义触发器图案 (全1白色方块)
        # 形状: (C, H, W)
        raw_trigger = torch.ones((self.num_channels, self.trigger_size, self.trigger_size))
        # 2. 归一化触发器 (使其与归一化后的图像数据分布一致)
        # trigger value = (1.0 - mean) / std
        norm_transform = transforms.Normalize(self.mean, self.std)
        self.trigger = norm_transform(raw_trigger)
        # 3. 定义触发器位置 (默认右下角)
        self.trigger_pos = (-self.trigger_size, -self.trigger_size)

    def _implant_trigger(self, image):
        """将触发器植入单张图像"""
        # image shape: (C, H, W)
        c, h, w = image.shape
        if c != self.num_channels:
            raise ValueError(f"图像通道数 {c} 与触发器通道数 {self.num_channels} 不匹配。")
        if self.trigger_size > h or self.trigger_size > w:
            raise ValueError("trigger_size 大于输入图像尺寸。")

        r_start, c_start = self.trigger_pos
        # 处理负索引切片逻辑
        r_end = r_start + self.trigger_size
        c_end = c_start + self.trigger_size

        r_end = None if r_end == 0 else r_end
        c_end = None if c_end == 0 else c_end

        # 覆盖像素
        image[..., r_start:r_end, c_start:c_end] = self.trigger
        return image

    def _get_target_label(self, original_label):
        """All-to-one: 被选中的样本统一改为 target_label。"""
        return self.target_label

    def _select_poison_indices(self, labels, train=True):
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
        """按 batch 在线执行 BadNets 投毒。"""
        poisoned_images = images.clone()
        poisoned_labels = labels.clone()
        poison_indices = self._select_poison_indices(poisoned_labels, train=train)

        for idx in poison_indices:
            original_label = poisoned_labels[idx].item()
            poisoned_images[idx] = self._implant_trigger(poisoned_images[idx])
            poisoned_labels[idx] = self._get_target_label(original_label)

        return poisoned_images, poisoned_labels

    def poison_dataset(self, dataset, train=True):
        """
        对数据集进行 BadNets 投毒。
        - train=True: 从恶意客户端全部非目标类样本中按 poisoning_ratio 随机采样投毒
        - train=False: 只保留已植入 trigger 的非目标类样本 (用于 all-to-one ASR 评估)
        """
        # 提取数据
        if isinstance(dataset, TensorDataset):
            images, labels = dataset.tensors
        else:
            loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            images, labels = next(iter(loader))

        eval_indices = self._select_poison_indices(labels, train=False) if not train else None
        poisoned_images, poisoned_labels = self.poison_batch(images, labels, train=train)

        if not train:
            # ASR 的分母应只包含原始非目标类样本；原本就是 target_label 的干净样本不能算攻击成功。
            if not eval_indices:
                print("Warning: No candidate samples found for BadNets ASR evaluation.")
                return TensorDataset(poisoned_images[:0], poisoned_labels[:0])
            eval_index_tensor = torch.as_tensor(eval_indices, dtype=torch.long)
            poisoned_images = poisoned_images[eval_index_tensor].clone()
            poisoned_labels = poisoned_labels[eval_index_tensor].clone()

        if len(poisoned_labels) == 0:
            print("Warning: No candidate samples found for BadNets poisoning.")
        return TensorDataset(poisoned_images, poisoned_labels)

    def get_poisoned_loader(self, dataset, batch_size=64, train=True, shuffle=True):
        poisoned_dataset = self.poison_dataset(dataset, train=train)
        return DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    print("Testing BadNets implementation")

    # Test all-to-one attack: random subset -> target 7
    labels = torch.tensor([1] * 10 + [3] * 10, dtype=torch.long)
    dataset = TensorDataset(torch.zeros(20, 1, 28, 28), labels)

    attacker = BadNetsAttack(
        dataset_name="mnist",
        target_label=7,
        poisoning_ratio=0.5,
    )
    poisoned = attacker.poison_dataset(dataset, train=True)

    poisoned_labels = poisoned.tensors[1]
    changed = (poisoned_labels != labels).sum().item()
    print(f"All-to-one changed: {changed}")
    if changed == int(len(labels) * 0.5):
        print("All-to-one BadNets check passed.")
    else:
        print("All-to-one BadNets check failed.")
