from __future__ import annotations

import torch

from .common import AggregationResult, aggregate_geometric_median, geometric_median


def median_aggregate(
    local_state_dicts,
    feature_matrix: torch.Tensor,
    num_malicious: int,
    global_state_dict,
    max_iters: int = 100,
    tol: float = 1e-5,
) -> AggregationResult:
    center = geometric_median(feature_matrix, max_iters=max_iters, tol=tol)
    distances = torch.norm(feature_matrix - center.unsqueeze(0), dim=1)
    top_f = min(num_malicious, len(local_state_dicts))
    predicted = torch.topk(distances, k=top_f).indices.tolist() if top_f > 0 else []
    aggregated_state = aggregate_geometric_median(
        local_state_dicts,
        reference_state_dict=global_state_dict,
        max_iters=max_iters,
        tol=tol,
    )
    return AggregationResult(
        aggregated_state=aggregated_state,
        predicted_malicious_ids=sorted(int(index) for index in predicted),
        selected_client_ids=list(range(len(local_state_dicts))),
        aux_scores={"distances": distances.tolist()},
    )
