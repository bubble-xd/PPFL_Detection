from __future__ import annotations

from .common import AggregationResult, aggregate_mean


def fedavg_aggregate(local_state_dicts, global_state_dict) -> AggregationResult:
    selected_client_ids = list(range(len(local_state_dicts)))
    # 客户端提交的是本地模型，因此服务端直接在模型参数空间做 FedAvg。
    aggregated_state = aggregate_mean(
        local_state_dicts,
        reference_state_dict=global_state_dict,
        selected_client_ids=selected_client_ids,
    )
    return AggregationResult(
        aggregated_state=aggregated_state,
        predicted_malicious_ids=[],
        selected_client_ids=selected_client_ids,
        aux_scores={},
    )
