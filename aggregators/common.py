from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor

from utils.state_dict import average_tensor_dicts, clone_state_dict, get_float_tensor_keys


@dataclass
class AggregationResult:
    aggregated_state: Dict[str, Tensor]
    predicted_malicious_ids: List[int]
    selected_client_ids: List[int]
    aux_scores: Dict[str, object] = field(default_factory=dict)


def pairwise_squared_distances(feature_matrix: Tensor) -> Tensor:
    return torch.cdist(feature_matrix, feature_matrix, p=2).pow(2)


def geometric_median(
    points: Tensor,
    max_iters: int = 100,
    tol: float = 1e-5,
) -> Tensor:
    if points.dim() != 2:
        raise ValueError("points must be a 2D tensor.")
    if points.size(0) == 1:
        return points[0].clone()

    current = points.mean(dim=0)
    for _ in range(max_iters):
        distances = torch.norm(points - current.unsqueeze(0), dim=1).clamp_min(1e-12)
        if torch.any(distances <= 1e-12):
            current = points[torch.argmin(distances)].clone()
            break
        weights = 1.0 / distances
        next_point = (weights.unsqueeze(1) * points).sum(dim=0) / weights.sum()
        if torch.norm(next_point - current).item() <= tol:
            current = next_point
            break
        current = next_point
    return current


def aggregate_geometric_median(
    local_state_dicts,
    reference_state_dict,
    max_iters: int = 100,
    tol: float = 1e-5,
) -> Dict[str, Tensor]:
    if not local_state_dicts:
        raise ValueError("local_state_dicts must not be empty.")
    float_keys = get_float_tensor_keys(reference_state_dict)

    def _state_dict_l2_distance(
        left_state: Dict[str, Tensor],
        right_state: Dict[str, Tensor],
    ) -> float:
        squared_sum = 0.0
        for key in float_keys:
            delta = (
                left_state[key].detach().to(device="cpu", dtype=torch.float32)
                - right_state[key].detach().to(device="cpu", dtype=torch.float32)
            )
            squared_sum += float(torch.sum(delta * delta).item())
        return squared_sum ** 0.5

    def _weighted_average_state_dicts(
        weights: torch.Tensor,
    ) -> Dict[str, Tensor]:
        total_weight = float(weights.sum().item())
        if total_weight <= 0.0:
            raise ValueError("weights must sum to a positive value.")

        float_key_set = set(float_keys)
        # 浮点参数后续会被加权均值覆盖，只复制非浮点状态以减少大模型聚合时的冗余内存写入。
        averaged = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in reference_state_dict.items()
            if key not in float_key_set
        }
        weighted_sums = {
            key: torch.zeros_like(reference_state_dict[key], device="cpu", dtype=torch.float32)
            for key in float_keys
        }

        # 加权均值同样按 state 顺序流式累加，避免磁盘后端为每个 key 反复加载整份模型。
        for state_index, state_dict in enumerate(local_state_dicts):
            state_weight = float(weights[state_index].item())
            for key in float_keys:
                weighted_sums[key].add_(
                    state_dict[key].detach().to(device="cpu", dtype=torch.float32),
                    alpha=state_weight,
                )

        for key in float_keys:
            mean_tensor = weighted_sums[key].div_(total_weight)
            averaged[key] = mean_tensor.to(
                device=reference_state_dict[key].device,
                dtype=reference_state_dict[key].dtype,
            )
        return averaged

    # 这里改成 state_dict 级别的流式 Weiszfeld：
    # 只保留“当前中心”和按 key 的累加器，避免把整批大模型 flatten 成超大矩阵。
    current = average_tensor_dicts(
        local_state_dicts,
        reference_state_dict=reference_state_dict,
    )
    for _ in range(max_iters):
        distances = torch.tensor(
            [_state_dict_l2_distance(local_state_dict, current) for local_state_dict in local_state_dicts],
            dtype=torch.float64,
        )
        if torch.any(distances <= 1e-12):
            current = clone_state_dict(
                local_state_dicts[int(torch.argmin(distances).item())],
                device="cpu",
            )
            break

        weights = 1.0 / distances
        next_state = _weighted_average_state_dicts(weights)
        if _state_dict_l2_distance(next_state, current) <= tol:
            current = next_state
            break
        current = next_state
    return current


def aggregate_mean(
    local_state_dicts,
    reference_state_dict,
    selected_client_ids: Optional[List[int]] = None,
) -> Dict[str, Tensor]:
    # 这里平均的是客户端提交的本地模型；非浮点字段继续沿用参考 state_dict 中的值。
    return average_tensor_dicts(
        local_state_dicts,
        selected_ids=selected_client_ids,
        reference_state_dict=reference_state_dict,
    )
