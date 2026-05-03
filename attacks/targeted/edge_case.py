# 参考文献: Attack of the Tails: Yes, You Really Can Backdoor Federated Learning (NeurIPS '20)
#  https://github.com/vio1etus/FLPoison/blob/main/attackers/edgecase.py

import os
import pickle
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader, TensorDataset
"""
Edge Case Attack (边缘样本攻击)
原理：利用数据分布中的边缘样本（Out-of-Distribution, OOD）作为后门触发器。
使用 ARDIS 和 Southwest Airline 数据集作为边缘样本
"""

# 定义数据路径
EDGE_CASE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "edge_case_data")
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _normalize_batch(images: torch.Tensor, mean, std) -> torch.Tensor:
    """对 (N, C, H, W) 张量做逐通道标准化。"""
    mean_t = torch.tensor(mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    return (images - mean_t) / std_t

# =============================================================================
# 1. 数据集定义 (ARDIS & Southwest Airline)
# =============================================================================

class ARDISDataset:
    """针对 MNIST 的 Edge Case 数据集 (ARDIS)"""
    def __init__(self, target_label=1, root=EDGE_CASE_DATA_DIR):
        self.root = root
        self.target_label = target_label
        self.source_label = 7
        self.data_path = os.path.join(self.root, "ARDIS")
        self.filenames = [
            'ARDIS_train_2828.csv', 'ARDIS_train_labels.csv',
            'ARDIS_test_2828.csv', 'ARDIS_test_labels.csv'
        ]
        self._load_data()

    def _load_data(self):
        if not all(os.path.exists(os.path.join(self.root, f)) for f in self.filenames):
            if all(os.path.exists(os.path.join(self.data_path, f)) for f in self.filenames):
                path = self.data_path
            else:
                raise FileNotFoundError(f"ARDIS 数据集文件缺失于 {self.root}")
        else:
            path = self.root

        def load_csv(name): 
            return torch.from_numpy(np.loadtxt(os.path.join(path, name), dtype='float32'))
        
        train_x, train_y = load_csv(self.filenames[0]), load_csv(self.filenames[1])
        test_x, test_y = load_csv(self.filenames[2]), load_csv(self.filenames[3])

        # 重塑为 (N, 1, 28, 28)，先映射到 [0,1]，再按 MNIST 统计量标准化
        self.train_images = train_x.reshape(-1, 1, 28, 28) / 255.0
        self.test_images = test_x.reshape(-1, 1, 28, 28) / 255.0
        self.train_images = _normalize_batch(self.train_images, MNIST_MEAN, MNIST_STD)
        self.test_images = _normalize_batch(self.test_images, MNIST_MEAN, MNIST_STD)
        
        # One-hot 转整数标签
        self.train_labels = torch.argmax(train_y, dim=1)
        self.test_labels = torch.argmax(test_y, dim=1)

        # 过滤出源类别 (7) 的样本作为 Edge Case；ASR 也应只评估同一语义来源的 held-out 样本。
        train_indices = self.train_labels == self.source_label
        test_indices = self.test_labels == self.source_label
        self.sampled_train_images = self.train_images[train_indices]
        self.sampled_test_images = self.test_images[test_indices]

    def get_poisoned_trainset(self):
        labels = torch.full((len(self.sampled_train_images),), self.target_label, dtype=torch.long)
        return self.sampled_train_images, labels

    def get_poisoned_testset(self):
        labels = torch.full((len(self.sampled_test_images),), self.target_label, dtype=torch.long)
        return TensorDataset(self.sampled_test_images, labels)


class SouthwestAirlineDataset:
    """针对 CIFAR 系列 RGB 数据的 Edge Case 数据集 (Southwest Airline)"""
    def __init__(
        self,
        target_label=9,
        root=EDGE_CASE_DATA_DIR,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    ):
        self.root = root
        self.target_label = target_label
        self.source_label = 0 # 飞机
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.filenames = ['southwest_images_new_train.pkl', 'southwest_images_new_test.pkl']
        self._load_data()

    def _load_data(self):
        paths = [os.path.join(self.root, f) for f in self.filenames]
        if not all(os.path.exists(p) for p in paths):
            raise FileNotFoundError(f"Southwest 数据集文件缺失于 {self.root}")

        with open(paths[0], 'rb') as f:
            self.train_images = pickle.load(f)
        with open(paths[1], 'rb') as f:
            self.test_images = pickle.load(f)

        # 格式转换: (N, H, W, C) -> (N, C, H, W), 归一化
        if self.train_images.shape[-1] == 3:
            self.train_images = np.transpose(self.train_images, (0, 3, 1, 2))
        if self.test_images.shape[-1] == 3:
            self.test_images = np.transpose(self.test_images, (0, 3, 1, 2))
            
        self.train_images = torch.from_numpy(self.train_images).float() / 255.0
        self.test_images = torch.from_numpy(self.test_images).float() / 255.0
        self.train_images = _normalize_batch(self.train_images, self.mean, self.std)
        self.test_images = _normalize_batch(self.test_images, self.mean, self.std)

    def get_poisoned_trainset(self):
        labels = torch.full((len(self.train_images),), self.target_label, dtype=torch.long)
        return self.train_images, labels

    def get_poisoned_testset(self):
        labels = torch.full((len(self.test_images),), self.target_label, dtype=torch.long)
        return TensorDataset(self.test_images, labels)

# 2. 核心攻击逻辑类

class EdgeCaseAttack:
    """
    Edge Case 后门攻击类
    集成了 Data Poisoning, PGD 投影和 Scaling Attack
    """
    def __init__(
        self, 
        dataset_name="mnist", 
        target_label=None, 
        poisoning_ratio=0.8,
        epsilon=0.25,
        projection_type="l_2",
        scaling_factor=50.0
    ):
        self.dataset_name = dataset_name.lower().replace("-", "")
        self.poisoning_ratio = poisoning_ratio
        self.epsilon = epsilon
        self.projection_type = projection_type
        self.scaling_factor = scaling_factor
        
        # 根据数据集初始化
        if "mnist" in self.dataset_name:
            self.target_label = target_label if target_label is not None else 1
            self.edge_obj = ARDISDataset(self.target_label)
        elif "cifar100" in self.dataset_name:
            # CIFAR100 没有专门的 edge-case 外部集，这里复用 Southwest RGB 样本；
            # 关键是改用 CIFAR100 统计量，让注入样本与 clean CIFAR100 输入同尺度。
            self.target_label = target_label if target_label is not None else 99
            self.edge_obj = SouthwestAirlineDataset(
                self.target_label,
                mean=CIFAR100_MEAN,
                std=CIFAR100_STD,
            )
        elif "cifar10" in self.dataset_name:
            self.target_label = target_label if target_label is not None else 9
            self.edge_obj = SouthwestAirlineDataset(
                self.target_label,
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            )
        else:
            raise ValueError(f"当前 Edge Case 实现仅支持 MNIST、CIFAR10 和 CIFAR100, 当前: {dataset_name}")

    def mix_train_loader(self, clean_loader: DataLoader) -> DataLoader:
        """数据投毒：混合边缘样本并生成新的 DataLoader"""
        # 1. 提取原始数据
        all_x, all_y = [], []
        for x, y in clean_loader:
            all_x.append(x)
            all_y.append(y)
        clean_x = torch.cat(all_x)
        clean_y = torch.cat(all_y)
        
        total_num = len(clean_x)
        poison_x, poison_y = self.edge_obj.get_poisoned_trainset()
        
        # 2. 计算混合数量 
        poison_num = int(total_num * self.poisoning_ratio)
        poison_num = min(poison_num, len(poison_x), total_num)
        benign_num = total_num - poison_num

        
        # 3. 采样并混合
        perm_benign = torch.randperm(total_num)[:benign_num]
        perm_poison = torch.randperm(len(poison_x))[:poison_num]
        
        mixed_x = torch.cat([clean_x[perm_benign], poison_x[perm_poison]])
        mixed_y = torch.cat([clean_y[perm_benign], poison_y[perm_poison]])
        
        # 4. 维度适配 (处理灰度/彩色或不同分辨率)
        if mixed_x.shape[1:] != clean_x.shape[1:]:
            mixed_x = torch.nn.functional.interpolate(mixed_x, size=clean_x.shape[2:])
            if mixed_x.shape[1] != clean_x.shape[1]:
                if clean_x.shape[1] == 1: # RGB to Gray
                    mixed_x = mixed_x.mean(dim=1, keepdim=True)
                else: # Gray to RGB
                    mixed_x = mixed_x.repeat(1, 3, 1, 1)

        return DataLoader(
            TensorDataset(mixed_x, mixed_y), 
            batch_size=clean_loader.batch_size, 
            shuffle=True
        )

    def poison_batch(self, images: torch.Tensor, labels: torch.Tensor):
        """按 batch 在线执行 edge-case 注入，替换部分 clean 样本。"""
        poisoned_images = images.clone()
        poisoned_labels = labels.clone()

        batch_size = len(poisoned_images)
        poison_num = int(batch_size * self.poisoning_ratio)
        if self.poisoning_ratio > 0 and poison_num == 0:
            poison_num = 1
        poison_num = min(poison_num, batch_size)
        if poison_num <= 0:
            return poisoned_images, poisoned_labels

        poison_x, poison_y = self.edge_obj.get_poisoned_trainset()
        if len(poison_x) == 0:
            return poisoned_images, poisoned_labels

        if len(poison_x) < poison_num:
            poison_indices = torch.randint(0, len(poison_x), (poison_num,))
        else:
            poison_indices = torch.randperm(len(poison_x))[:poison_num]

        target_indices = torch.randperm(batch_size)[:poison_num]
        sampled_x = poison_x[poison_indices]
        sampled_y = poison_y[poison_indices]

        if sampled_x.shape[1:] != poisoned_images.shape[1:]:
            sampled_x = torch.nn.functional.interpolate(
                sampled_x,
                size=poisoned_images.shape[2:],
            )
            if sampled_x.shape[1] != poisoned_images.shape[1]:
                if poisoned_images.shape[1] == 1:
                    sampled_x = sampled_x.mean(dim=1, keepdim=True)
                else:
                    sampled_x = sampled_x.repeat(1, 3, 1, 1)

        poisoned_images[target_indices] = sampled_x.to(
            device=poisoned_images.device,
            dtype=poisoned_images.dtype,
        )
        poisoned_labels[target_indices] = sampled_y.to(
            device=poisoned_labels.device,
            dtype=poisoned_labels.dtype,
        )
        return poisoned_images, poisoned_labels

    def get_poisoned_test_loader(self, batch_size=64) -> DataLoader:
        """获取全投毒测试集用于评估 ASR"""
        dataset = self.edge_obj.get_poisoned_testset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def apply_pgd(self, local_model: torch.nn.Module, global_model_state: dict, current_epoch: int):
        """模型投毒：执行 PGD 投影 (约束本地更新在全局模型的 epsilon 球内)"""
        local_state = local_model.state_dict()
        device = next(local_model.parameters()).device
        
        # 计算全局差异
        diff_list = []
        for key in local_state.keys():
            if key in global_model_state and local_state[key].dtype.is_floating_point:
                diff = local_state[key] - global_model_state[key].to(device)
                diff_list.append(diff.flatten())
        
        if not diff_list: return
        
        global_diff_vec = torch.cat(diff_list)
        
        if self.projection_type == "l_2":
            norm = torch.norm(global_diff_vec, p=2)
            if norm > self.epsilon:
                scale = self.epsilon / norm
                for key in local_state.keys():
                    if key in global_model_state and local_state[key].dtype.is_floating_point:
                        g_val = global_model_state[key].to(device)
                        local_state[key] = g_val + (local_state[key] - g_val) * scale
                local_model.load_state_dict(local_state)
                
        elif self.projection_type == "l_inf":
            for key in local_state.keys():
                if key in global_model_state and local_state[key].dtype.is_floating_point:
                    g_val = global_model_state[key].to(device)
                    diff = local_state[key] - g_val
                    local_state[key] = g_val + torch.clamp(diff, -self.epsilon, self.epsilon)
            local_model.load_state_dict(local_state)

# 3. 测试模块

if __name__ == "__main__":
    # 初始化攻击类 (针对 MNIST)
    attack = EdgeCaseAttack(dataset_name="mnist", poisoning_ratio=0.5, epsilon=0.5)
    
    # 1. 测试数据加载
    try:
        print(f"ARDIS Edge samples: {len(attack.edge_obj.sampled_train_images)}")
    except Exception as e:
        print(f"Load Error: {e}")

    # 2. 测试 PGD 逻辑
    model = torch.nn.Linear(10, 2)
    global_state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.weight.add_(2.0) # 制造一个大更新
        
    diff_before = torch.norm(model.weight - global_state['weight']).item()
    print(f"Before PGD diff norm: {diff_before:.4f}")
    
    attack.apply_pgd(model, global_state, current_epoch=0)
    
    diff_after = torch.norm(model.weight - global_state['weight']).item()
    print(f"After PGD (epsilon=0.5) diff norm: {diff_after:.4f}")
    
    if diff_after <= 0.5001:
        print("PGD implementation: SUCCESS")
    else:
        print("PGD implementation: FAILED")
