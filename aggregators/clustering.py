from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans

from .common import AggregationResult, aggregate_mean


def clustering_aggregate(
    local_state_dicts,
    feature_matrix,
    global_state_dict,
    random_state: int = 42,
    n_init: int = 10,
) -> AggregationResult:
    if len(local_state_dicts) < 2:
        aggregated_state = aggregate_mean(local_state_dicts, reference_state_dict=global_state_dict)
        return AggregationResult(
            aggregated_state=aggregated_state,
            predicted_malicious_ids=[],
            selected_client_ids=list(range(len(local_state_dicts))),
            aux_scores={},
        )

    features_np = feature_matrix.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(features_np)
    centers = kmeans.cluster_centers_

    cluster_members: Dict[int, List[int]] = {
        cluster_id: np.where(labels == cluster_id)[0].tolist()
        for cluster_id in range(2)
    }
    cluster_sizes = {cluster_id: len(indices) for cluster_id, indices in cluster_members.items()}
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda item: item[1])

    if abs(sorted_clusters[0][1] - sorted_clusters[1][1]) <= 1:
        global_center = features_np.mean(axis=0)
        cluster_distances = {
            cluster_id: float(np.linalg.norm(center - global_center))
            for cluster_id, center in enumerate(centers)
        }
        malicious_cluster = max(cluster_distances, key=cluster_distances.get)
    else:
        malicious_cluster = sorted_clusters[0][0]

    predicted_malicious_ids = sorted(int(index) for index in cluster_members[malicious_cluster])
    benign_ids = [
        client_id
        for client_id in range(len(local_state_dicts))
        if client_id not in predicted_malicious_ids
    ]
    if not benign_ids:
        benign_ids = list(range(len(local_state_dicts)))

    # 聚类模块只筛掉可疑客户端，最终直接对剩余客户端的本地模型做均值聚合。
    aggregated_state = aggregate_mean(
        local_state_dicts,
        reference_state_dict=global_state_dict,
        selected_client_ids=benign_ids,
    )
    return AggregationResult(
        aggregated_state=aggregated_state,
        predicted_malicious_ids=predicted_malicious_ids,
        selected_client_ids=benign_ids,
        aux_scores={"cluster_labels": labels.tolist()},
    )
