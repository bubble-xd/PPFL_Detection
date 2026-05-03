from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch

from utils.state_dict import flatten_tensor_dict, select_tensor_dict_by_prefixes

from .types import AttackLayerSummary, RoundLayerMetrics


def compute_population_variance(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values 不能为空。")
    mean_value = float(sum(float(value) for value in values) / len(values))
    return float(
        sum((float(value) - mean_value) ** 2 for value in values) / len(values)
    )


def zscore_layer_values(
    layer_values: Dict[str, float],
    epsilon: float,
) -> Dict[str, float]:
    if not layer_values:
        raise ValueError("layer_values 不能为空。")
    ordered_layers = list(layer_values.keys())
    values = [float(layer_values[layer_name]) for layer_name in ordered_layers]
    mean_value = float(sum(values) / len(values))
    std_value = math.sqrt(compute_population_variance(values))
    return {
        layer_name: float((layer_values[layer_name] - mean_value) / (std_value + float(epsilon)))
        for layer_name in ordered_layers
    }


def compute_adaptive_weights(
    magnitudes: Dict[str, float],
    cosine_distances: Dict[str, float],
    epsilon: float,
) -> Tuple[float, float]:
    magnitude_variance = compute_population_variance(list(magnitudes.values()))
    cosine_variance = compute_population_variance(list(cosine_distances.values()))
    denominator = magnitude_variance + cosine_variance + float(epsilon)
    alpha = float(magnitude_variance / denominator)
    beta = float(1.0 - alpha)
    return alpha, beta


def _extract_layer_vector(
    delta_dict: Dict[str, torch.Tensor],
    layer_prefix: str,
) -> torch.Tensor:
    selected = select_tensor_dict_by_prefixes(delta_dict, [layer_prefix])
    if not selected:
        raise ValueError(f"层前缀 {layer_prefix} 未匹配到任何浮点参数。")
    return flatten_tensor_dict(selected)


def _compute_layer_magnitude_and_cosine(
    benign_vector: torch.Tensor,
    malicious_vector: torch.Tensor,
    epsilon: float,
) -> Tuple[float, float]:
    layer_dim = int(benign_vector.numel())
    if layer_dim == 0:
        raise ValueError("候选层不能为空向量。")

    difference = malicious_vector - benign_vector
    magnitude = float(torch.norm(difference, p=2).item() / math.sqrt(layer_dim))

    benign_norm = float(torch.norm(benign_vector, p=2).item())
    malicious_norm = float(torch.norm(malicious_vector, p=2).item())
    # 双零更新时方向信息本身不存在，直接记 0 比硬算余弦距离更符合语义。
    if benign_norm <= float(epsilon) and malicious_norm <= float(epsilon):
        cosine_distance = 0.0
    else:
        cosine_similarity = float(
            torch.dot(benign_vector, malicious_vector).item()
            / (benign_norm * malicious_norm + float(epsilon))
        )
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
        cosine_distance = float(1.0 - cosine_similarity)
    return magnitude, cosine_distance


def compute_round_layer_metrics(
    candidate_layers: Sequence[str],
    benign_delta: Dict[str, torch.Tensor],
    malicious_delta: Dict[str, torch.Tensor],
    round_index: int,
    epsilon: float,
) -> RoundLayerMetrics:
    magnitudes: Dict[str, float] = {}
    cosine_distances: Dict[str, float] = {}

    for layer_prefix in candidate_layers:
        benign_vector = _extract_layer_vector(benign_delta, layer_prefix)
        malicious_vector = _extract_layer_vector(malicious_delta, layer_prefix)
        magnitude, cosine_distance = _compute_layer_magnitude_and_cosine(
            benign_vector=benign_vector,
            malicious_vector=malicious_vector,
            epsilon=epsilon,
        )
        magnitudes[layer_prefix] = magnitude
        cosine_distances[layer_prefix] = cosine_distance

    magnitude_zscores = zscore_layer_values(magnitudes, epsilon=epsilon)
    cosine_zscores = zscore_layer_values(cosine_distances, epsilon=epsilon)
    alpha, beta = compute_adaptive_weights(
        magnitudes=magnitudes,
        cosine_distances=cosine_distances,
        epsilon=epsilon,
    )
    combined_scores = {
        layer_prefix: float(
            alpha * magnitude_zscores[layer_prefix] + beta * cosine_zscores[layer_prefix]
        )
        for layer_prefix in candidate_layers
    }
    return RoundLayerMetrics(
        round_index=int(round_index),
        magnitudes=magnitudes,
        cosine_distances=cosine_distances,
        magnitude_zscores=magnitude_zscores,
        cosine_zscores=cosine_zscores,
        alpha=alpha,
        beta=beta,
        combined_scores=combined_scores,
    )


def summarize_attack_scores(
    attack_name: str,
    round_metrics: Sequence[RoundLayerMetrics],
    candidate_layers: Sequence[str],
) -> AttackLayerSummary:
    if not round_metrics:
        raise ValueError(f"attack={attack_name} 没有任何 round 指标。")

    layer_scores = {
        layer_prefix: float(
            sum(float(round_metric.combined_scores[layer_prefix]) for round_metric in round_metrics)
            / len(round_metrics)
        )
        for layer_prefix in candidate_layers
    }
    layer_order = {layer_prefix: index for index, layer_prefix in enumerate(candidate_layers)}
    top1_layer = max(
        candidate_layers,
        key=lambda layer_prefix: (layer_scores[layer_prefix], -layer_order[layer_prefix]),
    )
    return AttackLayerSummary(
        attack_name=str(attack_name).strip().lower(),
        round_metrics=list(round_metrics),
        layer_scores=layer_scores,
        top1_layer=top1_layer,
        top1_score=float(layer_scores[top1_layer]),
    )


def compute_consensus_scores(
    attack_summaries: Sequence[AttackLayerSummary],
    candidate_layers: Sequence[str],
) -> Dict[str, float]:
    return {
        layer_prefix: float(
            sum(max(0.0, float(summary.layer_scores[layer_prefix])) for summary in attack_summaries)
        )
        for layer_prefix in candidate_layers
    }
