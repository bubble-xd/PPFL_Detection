from __future__ import annotations

from typing import List

import torch

from .common import AggregationResult, aggregate_mean, pairwise_squared_distances


def _compute_krum_scores(feature_matrix: torch.Tensor, num_malicious: int) -> torch.Tensor:
    distances = pairwise_squared_distances(feature_matrix)
    num_clients = feature_matrix.size(0)
    if num_clients <= 1:
        return torch.zeros(num_clients, dtype=distances.dtype, device=distances.device)
    nearest_count = max(1, num_clients - num_malicious - 2)
    # 对角线置为 inf 后可以一次性 topk 出每个客户端最近邻，
    # 避免逐客户端 cat/sort 造成大量小张量分配。
    distances = distances.clone()
    distances.fill_diagonal_(float("inf"))
    nearest_values = torch.topk(
        distances,
        k=min(nearest_count, max(1, num_clients - 1)),
        dim=1,
        largest=False,
    ).values
    return nearest_values.sum(dim=1)


def krum_aggregate(local_state_dicts, feature_matrix, num_malicious: int, global_state_dict) -> AggregationResult:
    scores = _compute_krum_scores(feature_matrix, num_malicious=num_malicious)
    selected_client_id = int(torch.argmin(scores).item())
    top_f = min(num_malicious, len(local_state_dicts))
    predicted = torch.topk(scores, k=top_f).indices.tolist() if top_f > 0 else []
    # Krum 选中的是某个客户端本地模型；这里用参考 state_dict 回填完整模型结构。
    aggregated_state = aggregate_mean(
        local_state_dicts,
        reference_state_dict=global_state_dict,
        selected_client_ids=[selected_client_id],
    )
    return AggregationResult(
        aggregated_state=aggregated_state,
        predicted_malicious_ids=sorted(int(index) for index in predicted),
        selected_client_ids=[selected_client_id],
        aux_scores={"scores": scores.tolist()},
    )


def multi_krum_aggregate(
    local_state_dicts,
    feature_matrix,
    num_malicious: int,
    m: int,
    global_state_dict,
) -> AggregationResult:
    scores = _compute_krum_scores(feature_matrix, num_malicious=num_malicious)
    num_clients = len(local_state_dicts)
    top_f = min(num_malicious, num_clients)
    predicted = torch.topk(scores, k=top_f).indices.tolist() if top_f > 0 else []
    effective_m = max(1, min(int(m), num_clients))
    selected = torch.topk(-scores, k=effective_m).indices.tolist()
    aggregated_state = aggregate_mean(
        local_state_dicts,
        reference_state_dict=global_state_dict,
        selected_client_ids=selected,
    )
    return AggregationResult(
        aggregated_state=aggregated_state,
        predicted_malicious_ids=sorted(int(index) for index in predicted),
        selected_client_ids=sorted(int(index) for index in selected),
        aux_scores={"scores": scores.tolist()},
    )
