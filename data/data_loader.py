from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets as tv_datasets
from torchvision import transforms

from .fl_partition import partition_dataset


DATASET_INFO: Dict[str, Dict[str, object]] = {
    "mnist": {
        "num_classes": 10,
        "input_channels": 1,
        "image_size": 28,
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "cifar10": {
        "num_classes": 10,
        "input_channels": 3,
        "image_size": 32,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "num_classes": 100,
        "input_channels": 3,
        "image_size": 32,
        # CIFAR100 与 CIFAR10 同为 32x32 RGB，但统计量不同；
        # 单独配置可避免触发器和 clean 数据落在不一致的归一化空间。
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
}


@dataclass
class FederatedDataBundle:
    dataset_name: str
    train_dataset: torch.utils.data.Dataset
    test_dataset: TensorDataset
    client_datasets: Dict[int, TensorDataset]
    test_loader: DataLoader
    dataset_info: Dict[str, object]
    partition_info: Dict[str, object]


def _get_dataset_info(dataset_name: str) -> Dict[str, object]:
    normalized_name = str(dataset_name).strip().lower()
    if normalized_name not in DATASET_INFO:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dict(DATASET_INFO[normalized_name])


def _build_transform(dataset_name: str) -> transforms.Compose:
    info = _get_dataset_info(dataset_name)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(info["mean"], info["std"]),
        ]
    )


def _load_datasets(
    dataset_name: str,
    root: str = "data",
):
    normalized_name = str(dataset_name).strip().lower()
    transform = _build_transform(normalized_name)
    if normalized_name == "mnist":
        train_dataset = tv_datasets.MNIST(root=root, train=True, transform=transform, download=False)
        test_dataset = tv_datasets.MNIST(root=root, train=False, transform=transform, download=False)
        return train_dataset, test_dataset
    if normalized_name == "cifar10":
        train_dataset = tv_datasets.CIFAR10(root=root, train=True, transform=transform, download=False)
        test_dataset = tv_datasets.CIFAR10(root=root, train=False, transform=transform, download=False)
        return train_dataset, test_dataset
    if normalized_name == "cifar100":
        train_dataset = tv_datasets.CIFAR100(root=root, train=True, transform=transform, download=False)
        test_dataset = tv_datasets.CIFAR100(root=root, train=False, transform=transform, download=False)
        return train_dataset, test_dataset
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _extract_targets(dataset) -> torch.Tensor:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        raise AttributeError("Dataset does not expose targets or labels.")

    if isinstance(targets, list):
        return torch.tensor(targets, dtype=torch.long)
    return torch.as_tensor(targets, dtype=torch.long)


def materialize_dataset(dataset) -> TensorDataset:
    if isinstance(dataset, TensorDataset):
        images, labels = dataset.tensors
        return TensorDataset(images.clone(), labels.clone())

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(loader))
    return TensorDataset(images, labels)


def _materialize_subset(dataset, indices: List[int]) -> TensorDataset:
    subset = Subset(dataset, indices)
    return materialize_dataset(subset)


def build_federated_datasets(
    dataset_name: str,
    num_clients: int,
    partition: Optional[Dict[str, object]] = None,
    batch_size: int = 64,
    root: str = "data",
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> FederatedDataBundle:
    train_dataset, raw_test_dataset = _load_datasets(dataset_name, root=root)
    targets = _extract_targets(train_dataset)
    partition_info = partition_dataset(
        labels=targets,
        num_clients=num_clients,
        partition=partition or {"type": "iid"},
    )
    client_datasets = {
        client_id: _materialize_subset(train_dataset, indices)
        for client_id, indices in partition_info["client_indices"].items()
    }
    test_dataset = materialize_dataset(raw_test_dataset)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(pin_memory),
    }
    if int(loader_kwargs["num_workers"]) > 0:
        # 测试集同样常驻内存，worker 只负责批次搬运；是否开启由配置决定。
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    test_loader = DataLoader(test_dataset, **loader_kwargs)
    return FederatedDataBundle(
        dataset_name=str(dataset_name).strip().lower(),
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        client_datasets=client_datasets,
        test_loader=test_loader,
        dataset_info=_get_dataset_info(dataset_name),
        partition_info=partition_info,
    )
