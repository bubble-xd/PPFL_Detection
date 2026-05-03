from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def partition_iid(
    num_samples: int,
    num_clients: int,
    seed: int,
) -> Dict[int, List[int]]:
    generator = _rng(seed)
    indices = np.arange(num_samples)
    generator.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return {
        int(client_id): [int(index) for index in split.tolist()]
        for client_id, split in enumerate(splits)
    }


def partition_dirichlet(
    labels: torch.Tensor,
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 10,
    max_attempts: int = 50,
) -> Dict[int, List[int]]:
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")

    labels_np = labels.detach().cpu().numpy()
    generator = _rng(seed)
    num_classes = int(labels.max().item()) + 1

    for _ in range(max_attempts):
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for class_id in range(num_classes):
            class_indices = np.where(labels_np == class_id)[0]
            generator.shuffle(class_indices)
            proportions = generator.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            cut_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            split_indices = np.split(class_indices, cut_points)
            for client_id, subset in enumerate(split_indices):
                client_indices[client_id].extend(int(index) for index in subset.tolist())

        sizes = [len(indices) for indices in client_indices]
        if min(sizes) >= min_size:
            return {
                int(client_id): sorted(indices)
                for client_id, indices in enumerate(client_indices)
            }

    return partition_iid(num_samples=len(labels), num_clients=num_clients, seed=seed)


def partition_dataset(
    labels: torch.Tensor,
    num_clients: int,
    partition: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    partition = dict(partition or {"type": "iid"})
    partition_type = str(partition.get("type", "iid")).strip().lower()
    seed = int(partition.get("seed", 42))

    if partition_type == "iid":
        client_indices = partition_iid(
            num_samples=len(labels),
            num_clients=num_clients,
            seed=seed,
        )
        return {
            "type": "iid",
            "alpha": None,
            "seed": seed,
            "client_indices": client_indices,
        }

    if partition_type == "dirichlet":
        alpha = float(partition.get("alpha", 0.5))
        min_size = int(partition.get("min_size", 10))
        client_indices = partition_dirichlet(
            labels=labels,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            min_size=min_size,
        )
        return {
            "type": "dirichlet",
            "alpha": alpha,
            "seed": seed,
            "client_indices": client_indices,
        }

    raise ValueError(f"Unsupported partition type: {partition_type}")
