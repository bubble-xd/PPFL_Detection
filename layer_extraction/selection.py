from __future__ import annotations

from typing import Dict, Sequence

from .scoring import compute_consensus_scores
from .types import AttackLayerSummary, SelectionResult


def select_layers(
    model_name: str,
    dataset_name: str,
    partition_name: str,
    num_rounds: int,
    candidate_layers: Sequence[str],
    attack_summaries: Sequence[AttackLayerSummary],
    k: int,
    weighting_mode: str,
) -> SelectionResult:
    if k <= 0:
        raise ValueError("K 必须为正数。")

    layer_order = {layer_prefix: index for index, layer_prefix in enumerate(candidate_layers)}
    consensus_scores = compute_consensus_scores(
        attack_summaries=attack_summaries,
        candidate_layers=candidate_layers,
    )
    per_attack_scores = {
        summary.attack_name: dict(summary.layer_scores)
        for summary in attack_summaries
    }

    top1_by_attack: Dict[str, Dict[str, object]] = {}
    for summary in attack_summaries:
        top1_by_attack[summary.attack_name] = {
            "layer": summary.top1_layer,
            "score": float(summary.top1_score),
            "selected_in_final_set": False,
        }

    # 最终选层直接以跨攻击共识分数为准；
    # Top-1 只保留为解释性信息，不再参与最终截断逻辑。
    selected_layers = sorted(
        candidate_layers,
        key=lambda layer_prefix: (
            -consensus_scores[layer_prefix],
            layer_order[layer_prefix],
        ),
    )
    selected_layers = selected_layers[: min(k, len(candidate_layers))]

    dropped_top1_by_attack: Dict[str, Dict[str, object]] = {}
    for attack_name, top1_info in top1_by_attack.items():
        is_selected = str(top1_info["layer"]) in selected_layers
        top1_info["selected_in_final_set"] = is_selected
        if not is_selected:
            dropped_top1_by_attack[attack_name] = {
                "layer": top1_info["layer"],
                "score": float(top1_info["score"]),
            }

    return SelectionResult(
        model=str(model_name).strip().lower(),
        dataset=str(dataset_name).strip().lower(),
        partition=str(partition_name).strip().lower(),
        attacks=[summary.attack_name for summary in attack_summaries],
        num_rounds=int(num_rounds),
        k=int(k),
        candidate_layers=list(candidate_layers),
        selected_layers=selected_layers,
        top1_by_attack=top1_by_attack,
        per_attack_scores=per_attack_scores,
        consensus_scores=consensus_scores,
        weighting_mode=str(weighting_mode),
        config_key_layer_map_entry={str(model_name).strip().lower(): list(selected_layers)},
        dropped_top1_by_attack=dropped_top1_by_attack,
    )
