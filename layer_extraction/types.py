from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class RoundLayerMetrics:
    round_index: int
    magnitudes: Dict[str, float]
    cosine_distances: Dict[str, float]
    magnitude_zscores: Dict[str, float]
    cosine_zscores: Dict[str, float]
    alpha: float
    beta: float
    combined_scores: Dict[str, float]


@dataclass
class AttackLayerSummary:
    attack_name: str
    round_metrics: List[RoundLayerMetrics]
    layer_scores: Dict[str, float]
    top1_layer: str
    top1_score: float


@dataclass
class SelectionResult:
    model: str
    dataset: str
    partition: str
    attacks: List[str]
    num_rounds: int
    k: int
    candidate_layers: List[str]
    selected_layers: List[str]
    top1_by_attack: Dict[str, Dict[str, object]]
    per_attack_scores: Dict[str, Dict[str, float]]
    consensus_scores: Dict[str, float]
    weighting_mode: str
    config_key_layer_map_entry: Dict[str, List[str]]
    dropped_top1_by_attack: Dict[str, Dict[str, object]] = field(default_factory=dict)
