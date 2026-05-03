from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from typing import Dict, Sequence

from .types import AttackLayerSummary, SelectionResult


def _slugify(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace(".", "p")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text.strip("_") or "na"


def create_output_dir(
    results_root: str,
    output_prefix: str,
    model_name: str,
    dataset_name: str,
    partition_name: str,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory_name = (
        f"{_slugify(output_prefix)}_{timestamp}_{_slugify(model_name)}_"
        f"{_slugify(dataset_name)}_{_slugify(partition_name)}"
    )
    output_dir = os.path.join(results_root, directory_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def export_selection_artifacts(
    output_dir: str,
    result: SelectionResult,
    attack_summaries: Sequence[AttackLayerSummary],
) -> Dict[str, str]:
    selection_path = os.path.join(output_dir, "selection.json")
    layer_scores_path = os.path.join(output_dir, "layer_scores.csv")

    # 选择结果除了最终层集合，也要保留每轮 alpha/beta，方便后续核查权重是否真的自适应。
    selection_payload = {
        "model": result.model,
        "dataset": result.dataset,
        "partition": result.partition,
        "attacks": result.attacks,
        "num_rounds": result.num_rounds,
        "k": result.k,
        "candidate_layers": result.candidate_layers,
        "selected_layers": result.selected_layers,
        "top1_by_attack": result.top1_by_attack,
        "dropped_top1_by_attack": result.dropped_top1_by_attack,
        "per_attack_scores": result.per_attack_scores,
        "consensus_scores": result.consensus_scores,
        "weighting_mode": result.weighting_mode,
        "config_key_layer_map_entry": result.config_key_layer_map_entry,
        "round_weight_trace": {
            summary.attack_name: [
                {
                    "round": round_metric.round_index,
                    "alpha": float(round_metric.alpha),
                    "beta": float(round_metric.beta),
                }
                for round_metric in summary.round_metrics
            ]
            for summary in attack_summaries
        },
    }
    with open(selection_path, "w", encoding="utf-8") as handle:
        json.dump(selection_payload, handle, indent=2, ensure_ascii=False)

    fieldnames = ["layer", "selected", "selection_rank", "consensus_score", "top1_attacks"]
    fieldnames.extend([f"{summary.attack_name}_score" for summary in attack_summaries])
    selection_rank_map = {
        layer_prefix: index + 1
        for index, layer_prefix in enumerate(result.selected_layers)
    }
    top1_attack_map = {
        layer_prefix: [
            summary.attack_name
            for summary in attack_summaries
            if summary.top1_layer == layer_prefix
        ]
        for layer_prefix in result.candidate_layers
    }
    with open(layer_scores_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer_prefix in result.candidate_layers:
            row = {
                "layer": layer_prefix,
                "selected": "yes" if layer_prefix in selection_rank_map else "no",
                "selection_rank": selection_rank_map.get(layer_prefix, ""),
                "consensus_score": result.consensus_scores[layer_prefix],
                "top1_attacks": ",".join(top1_attack_map[layer_prefix]),
            }
            for summary in attack_summaries:
                row[f"{summary.attack_name}_score"] = summary.layer_scores[layer_prefix]
            writer.writerow(row)

    return {
        "selection_path": selection_path,
        "layer_scores_path": layer_scores_path,
    }
