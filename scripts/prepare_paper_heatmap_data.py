#!/usr/bin/env python3
"""Prepare existing cosine heatmap CSVs for paper plotting."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_heatmap_data"
POISON_RATE_DIR = "poison_rate_0p2"
HEATMAP_ROOT = Path("cosine_heatmaps") / "shared_global_base"
ROUND_FILE_PATTERN = re.compile(r"round_(\d+)_heatmap_data\.csv$")


@dataclass(frozen=True)
class ExperimentSpec:
    model_key: str
    model_display: str
    dataset: str
    source_experiment: str


@dataclass(frozen=True)
class AttackSpec:
    attack_key: str
    attack_display: str
    attack_mode: str
    attack_dir: str
    mode_row: int
    attack_order: int


@dataclass(frozen=True)
class FeatureSpec:
    feature_view: str
    feature_display: str
    feature_mode: str
    feature_order: int


EXPERIMENTS: Sequence[ExperimentSpec] = (
    ExperimentSpec(
        model_key="lenet5",
        model_display="LeNet-5",
        dataset="MNIST",
        source_experiment="20260427_120206_lenet5_mnist_iid_lr0p01",
    ),
    ExperimentSpec(
        model_key="resnet20",
        model_display="ResNet20",
        dataset="CIFAR-10",
        source_experiment="20260429_173501_resnet20_cifar10_iid_lr0p01",
    ),
    ExperimentSpec(
        model_key="resnet34",
        model_display="ResNet34",
        dataset="CIFAR-100",
        source_experiment="20260502_223505_resnet34_cifar100_iid_lr0p01",
    ),
)


ATTACKS: Sequence[AttackSpec] = (
    AttackSpec(
        attack_key="badnets",
        attack_display="BadNets",
        attack_mode="targeted",
        attack_dir="badnets_targeted",
        mode_row=1,
        attack_order=1,
    ),
    AttackSpec(
        attack_key="label_flipping_targeted",
        attack_display="Label Flipping",
        attack_mode="targeted",
        attack_dir="label_flipping_targeted_targeted",
        mode_row=1,
        attack_order=2,
    ),
    AttackSpec(
        attack_key="noise",
        attack_display="Noise",
        attack_mode="untargeted",
        attack_dir="additive_noise_untargeted",
        mode_row=2,
        attack_order=1,
    ),
    AttackSpec(
        attack_key="label_flipping_untargeted",
        attack_display="Label Flipping",
        attack_mode="untargeted",
        attack_dir="label_flipping_untargeted_untargeted",
        mode_row=2,
        attack_order=2,
    ),
)


FEATURES: Sequence[FeatureSpec] = (
    FeatureSpec(
        feature_view="full",
        feature_display="Full",
        feature_mode="raw_full",
        feature_order=1,
    ),
    FeatureSpec(
        feature_view="selected",
        feature_display="Selected",
        feature_mode="selected_layers_balanced_projected",
        feature_order=2,
    ),
)


METRIC_COLUMNS: Sequence[str] = (
    "bm_gap",
    "mean_benign_benign_cosine",
    "mean_benign_malicious_cosine",
    "mean_malicious_malicious_cosine",
    "silhouette_score",
)


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _available_round_files(attack_root: Path) -> List[Tuple[int, Path]]:
    round_files: List[Tuple[int, Path]] = []
    for csv_path in attack_root.glob("round_*_heatmap_data.csv"):
        match = ROUND_FILE_PATTERN.match(csv_path.name)
        if match is None:
            continue
        round_files.append((int(match.group(1)), csv_path))
    return sorted(round_files, key=lambda item: item[0])


def _select_final_round_csv(attack_root: Path) -> Tuple[int, List[int], Path]:
    round_files = _available_round_files(attack_root)
    if not round_files:
        raise FileNotFoundError(f"No round heatmap CSV found under {attack_root}")
    final_round, final_csv = round_files[-1]
    available_rounds = [round_idx for round_idx, _ in round_files]
    return final_round, available_rounds, final_csv


def _split_feature_rows(rows: Sequence[Dict[str, str]], feature_mode: str) -> List[Dict[str, str]]:
    feature_rows = [row for row in rows if row.get("feature_mode") == feature_mode]
    if not feature_rows:
        raise ValueError(f"Missing feature_mode={feature_mode!r}")
    return feature_rows


def _client_order(feature_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    by_position: Dict[int, Dict[str, str]] = {}
    for row in feature_rows:
        position = int(row["row_position"])
        by_position.setdefault(
            position,
            {
                "row_position": position,
                "client_id": int(row["row_client_id"]),
                "client_label": row["row_label"],
                "is_malicious": int(row["row_is_malicious"]),
            },
        )
    return [by_position[position] for position in sorted(by_position)]


def _feature_metrics(feature_rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    first_row = feature_rows[0]
    return {column: first_row.get(column, "") for column in METRIC_COLUMNS}


def _matrix_values(feature_rows: Sequence[Dict[str, str]]) -> Dict[Tuple[int, int], str]:
    values: Dict[Tuple[int, int], str] = {}
    for row in feature_rows:
        key = (int(row["row_position"]), int(row["col_position"]))
        values[key] = row["cosine_similarity"]
    return values


def _write_matrix_csv(
    matrix_path: Path,
    feature_rows: Sequence[Dict[str, str]],
    client_rows: Sequence[Dict[str, str]],
) -> None:
    matrix_values = _matrix_values(feature_rows)
    label_by_position = {int(row["row_position"]): str(row["client_label"]) for row in client_rows}
    fieldnames = [
        "row_position",
        "row_client_id",
        "row_label",
        "row_is_malicious",
        *[label_by_position[position] for position in sorted(label_by_position)],
    ]

    matrix_rows: List[Dict[str, object]] = []
    for client in client_rows:
        row_position = int(client["row_position"])
        matrix_row: Dict[str, object] = {
            "row_position": row_position,
            "row_client_id": int(client["client_id"]),
            "row_label": str(client["client_label"]),
            "row_is_malicious": int(client["is_malicious"]),
        }
        for col_position in sorted(label_by_position):
            # 宽矩阵直接面向绘图；行列标签和恶意标记保留，后续不用回查原始 CSV。
            row_key = label_by_position[col_position]
            row_matrix_key = (row_position, col_position)
            if row_matrix_key not in matrix_values:
                raise ValueError(f"Incomplete matrix: missing cell {row_matrix_key}")
            matrix_row[row_key] = matrix_values[row_matrix_key]
        matrix_rows.append(matrix_row)

    _write_csv(matrix_path, fieldnames, matrix_rows)


def _long_rows(
    source_rows: Sequence[Dict[str, str]],
    base: Dict[str, object],
    feature: FeatureSpec,
    matrix_csv: Path,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in source_rows:
        if row.get("feature_mode") != feature.feature_mode:
            continue
        rows.append(
            {
                **base,
                "feature_view": feature.feature_view,
                "feature_display": feature.feature_display,
                "feature_mode": feature.feature_mode,
                "matrix_csv": _relative_to_project(matrix_csv),
                "row_position": int(row["row_position"]),
                "col_position": int(row["col_position"]),
                "row_client_id": int(row["row_client_id"]),
                "col_client_id": int(row["col_client_id"]),
                "row_label": row["row_label"],
                "col_label": row["col_label"],
                "row_is_malicious": int(row["row_is_malicious"]),
                "col_is_malicious": int(row["col_is_malicious"]),
                "cosine_similarity": row["cosine_similarity"],
            }
        )
    return rows


def _manifest_row(
    experiment: ExperimentSpec,
    attack: AttackSpec,
    feature: FeatureSpec,
    final_round: int,
    available_rounds: Sequence[int],
    source_csv: Path,
    matrix_csv: Path,
    feature_rows: Sequence[Dict[str, str]],
) -> Dict[str, object]:
    metrics = _feature_metrics(feature_rows)
    figure_col = (attack.attack_order - 1) * 2 + feature.feature_order
    client_rows = _client_order(feature_rows)
    malicious_count = sum(int(row["is_malicious"]) for row in client_rows)
    return {
        "model_key": experiment.model_key,
        "model_display": experiment.model_display,
        "dataset": experiment.dataset,
        "source_experiment": experiment.source_experiment,
        "poison_rate": "0.2",
        "attack_mode": attack.attack_mode,
        "attack_key": attack.attack_key,
        "attack_display": attack.attack_display,
        "source_attack_dir": attack.attack_dir,
        "feature_view": feature.feature_view,
        "feature_display": feature.feature_display,
        "feature_mode": feature.feature_mode,
        "figure_row": attack.mode_row,
        "figure_col": figure_col,
        "final_saved_round": final_round,
        "available_rounds": " ".join(f"{round_idx:03d}" for round_idx in available_rounds),
        "source_csv": _relative_to_project(source_csv),
        "matrix_csv": _relative_to_project(matrix_csv),
        "num_clients": len(client_rows),
        "num_benign_clients": len(client_rows) - malicious_count,
        "num_malicious_clients": malicious_count,
        **metrics,
    }


def _pair_summary_rows(manifest_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, object]]] = {}
    for row in manifest_rows:
        key = (str(row["model_key"]), str(row["attack_key"]))
        grouped.setdefault(key, {})[str(row["feature_view"])] = row

    model_rank = {spec.model_key: rank for rank, spec in enumerate(EXPERIMENTS)}
    attack_rank = {spec.attack_key: rank for rank, spec in enumerate(ATTACKS)}
    summary_rows: List[Dict[str, object]] = []
    for key in grouped:
        feature_rows = grouped[key]
        full = feature_rows.get("full")
        selected = feature_rows.get("selected")
        if full is None or selected is None:
            continue
        full_gap = float(full["bm_gap"])
        selected_gap = float(selected["bm_gap"])
        summary_rows.append(
            {
                "model_key": full["model_key"],
                "model_display": full["model_display"],
                "dataset": full["dataset"],
                "source_experiment": full["source_experiment"],
                "poison_rate": full["poison_rate"],
                "attack_mode": full["attack_mode"],
                "attack_key": full["attack_key"],
                "attack_display": full["attack_display"],
                "figure_row": full["figure_row"],
                "figure_col_full": full["figure_col"],
                "figure_col_selected": selected["figure_col"],
                "final_saved_round": full["final_saved_round"],
                "available_rounds": full["available_rounds"],
                "full_bm_gap": f"{full_gap:.6f}",
                "selected_bm_gap": f"{selected_gap:.6f}",
                "bm_gap_delta": f"{selected_gap - full_gap:.6f}",
                "full_mean_benign_benign_cosine": full["mean_benign_benign_cosine"],
                "full_mean_benign_malicious_cosine": full["mean_benign_malicious_cosine"],
                "selected_mean_benign_benign_cosine": selected["mean_benign_benign_cosine"],
                "selected_mean_benign_malicious_cosine": selected["mean_benign_malicious_cosine"],
                "full_matrix_csv": full["matrix_csv"],
                "selected_matrix_csv": selected["matrix_csv"],
            }
        )

    # 摘要表按论文中的 2x4 面板顺序排序，避免绘图时再手动重排。
    return sorted(
        summary_rows,
        key=lambda row: (
            model_rank[str(row["model_key"])],
            int(row["figure_row"]),
            attack_rank[str(row["attack_key"])],
        ),
    )


def _write_readme(output_dir: Path, manifest_rows: Sequence[Dict[str, object]]) -> None:
    source_experiments = sorted({str(row["source_experiment"]) for row in manifest_rows})
    readme = {
        "purpose": "Paper heatmap plotting data prepared from existing cosine heatmap CSVs.",
        "data_scope": {
            "poison_rate": "0.2",
            "round_policy": "Use the final saved heatmap CSV for each experiment/attack.",
            "full_feature_mode": "raw_full",
            "selected_feature_mode": "selected_layers_balanced_projected",
            "source_experiments": source_experiments,
        },
        "files": {
            "manifest.csv": "One row per subplot panel.",
            "heatmap_long.csv": "Long-form matrix cells for direct pivot/plotting.",
            "bm_gap_pair_summary.csv": "Full vs Selected BM-Gap summary per model and attack.",
            "matrices/": "Wide matrix CSVs, one per model/attack/feature panel.",
        },
        "layout": {
            "figure_row_1": "targeted: BadNets Full, BadNets Selected, Label Flipping Full, Label Flipping Selected",
            "figure_row_2": "untargeted: Noise Full, Noise Selected, Label Flipping Full, Label Flipping Selected",
        },
    }
    (output_dir / "README.json").write_text(
        json.dumps(readme, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare_data(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, object]] = []
    all_long_rows: List[Dict[str, object]] = []

    for experiment in EXPERIMENTS:
        experiment_root = PROJECT_ROOT / "results" / experiment.source_experiment / POISON_RATE_DIR
        for attack in ATTACKS:
            attack_root = experiment_root / HEATMAP_ROOT / attack.attack_dir
            final_round, available_rounds, source_csv = _select_final_round_csv(attack_root)
            source_rows = _read_csv_rows(source_csv)

            for feature in FEATURES:
                feature_rows = _split_feature_rows(source_rows, feature.feature_mode)
                matrix_csv = (
                    output_dir
                    / "matrices"
                    / experiment.model_key
                    / attack.attack_mode
                    / attack.attack_key
                    / f"{feature.feature_view}_round_{final_round:03d}_matrix.csv"
                )

                # 先把每个 panel 的矩阵单独落盘，绘图时可按 manifest 精确定位。
                client_rows = _client_order(feature_rows)
                _write_matrix_csv(matrix_csv, feature_rows, client_rows)

                base = {
                    "model_key": experiment.model_key,
                    "model_display": experiment.model_display,
                    "dataset": experiment.dataset,
                    "source_experiment": experiment.source_experiment,
                    "poison_rate": "0.2",
                    "attack_mode": attack.attack_mode,
                    "attack_key": attack.attack_key,
                    "attack_display": attack.attack_display,
                    "source_attack_dir": attack.attack_dir,
                    "figure_row": attack.mode_row,
                    "figure_col": (attack.attack_order - 1) * 2 + feature.feature_order,
                    "final_saved_round": final_round,
                    "available_rounds": " ".join(f"{round_idx:03d}" for round_idx in available_rounds),
                    "source_csv": _relative_to_project(source_csv),
                }
                all_long_rows.extend(_long_rows(source_rows, base, feature, matrix_csv))
                manifest_rows.append(
                    _manifest_row(
                        experiment=experiment,
                        attack=attack,
                        feature=feature,
                        final_round=final_round,
                        available_rounds=available_rounds,
                        source_csv=source_csv,
                        matrix_csv=matrix_csv,
                        feature_rows=feature_rows,
                    )
                )

    manifest_fields = [
        "model_key",
        "model_display",
        "dataset",
        "source_experiment",
        "poison_rate",
        "attack_mode",
        "attack_key",
        "attack_display",
        "source_attack_dir",
        "feature_view",
        "feature_display",
        "feature_mode",
        "figure_row",
        "figure_col",
        "final_saved_round",
        "available_rounds",
        "source_csv",
        "matrix_csv",
        "num_clients",
        "num_benign_clients",
        "num_malicious_clients",
        *METRIC_COLUMNS,
    ]
    _write_csv(output_dir / "manifest.csv", manifest_fields, manifest_rows)

    long_fields = [
        "model_key",
        "model_display",
        "dataset",
        "source_experiment",
        "poison_rate",
        "attack_mode",
        "attack_key",
        "attack_display",
        "source_attack_dir",
        "feature_view",
        "feature_display",
        "feature_mode",
        "figure_row",
        "figure_col",
        "final_saved_round",
        "available_rounds",
        "source_csv",
        "matrix_csv",
        "row_position",
        "col_position",
        "row_client_id",
        "col_client_id",
        "row_label",
        "col_label",
        "row_is_malicious",
        "col_is_malicious",
        "cosine_similarity",
    ]
    _write_csv(output_dir / "heatmap_long.csv", long_fields, all_long_rows)

    pair_fields = [
        "model_key",
        "model_display",
        "dataset",
        "source_experiment",
        "poison_rate",
        "attack_mode",
        "attack_key",
        "attack_display",
        "figure_row",
        "figure_col_full",
        "figure_col_selected",
        "final_saved_round",
        "available_rounds",
        "full_bm_gap",
        "selected_bm_gap",
        "bm_gap_delta",
        "full_mean_benign_benign_cosine",
        "full_mean_benign_malicious_cosine",
        "selected_mean_benign_benign_cosine",
        "selected_mean_benign_malicious_cosine",
        "full_matrix_csv",
        "selected_matrix_csv",
    ]
    _write_csv(output_dir / "bm_gap_pair_summary.csv", pair_fields, _pair_summary_rows(manifest_rows))
    _write_readme(output_dir, manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write prepared paper heatmap data.",
    )
    args = parser.parse_args()
    prepare_data(args.output_dir.resolve())
    print(f"Prepared paper heatmap data under {_relative_to_project(args.output_dir.resolve())}")


if __name__ == "__main__":
    main()
