from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# 沙箱环境里 home 配置目录可能不可写，显式把 Matplotlib 缓存放到 /tmp。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


@dataclass(frozen=True)
class RankedLayerScore:
    rank: int
    layer: str
    score: float
    normalized_rank: float
    normalized_score: float
    elbow_distance: float
    next_gap: Optional[float]


@dataclass(frozen=True)
class AutoKResult:
    source_path: str
    model: str
    dataset: str
    partition: str
    existing_k: Optional[int]
    recommended_k: int
    selected_layers: List[str]
    existing_selected_layers: List[str]
    method: str
    min_k: int
    max_k: int
    max_k_ratio: Optional[float]
    ranked_layers: List[RankedLayerScore]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_selection_payload(selection_path: str | Path) -> Dict[str, Any]:
    path = Path(selection_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 不是合法的 selection.json 对象。")
    return payload


def find_selection_paths(results_root: str | Path) -> List[Path]:
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"results_root 不存在: {root}")
    if root.is_file():
        return [root] if root.name == "selection.json" else []

    # 只消费已有离线结果，避免误触发任何重新训练流程。
    direct = root / "selection.json"
    if direct.exists():
        return [direct]
    return sorted(path for path in root.glob("*/selection.json") if path.is_file())


def find_score_paths(results_root: str | Path) -> List[Path]:
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"results_root 不存在: {root}")
    if root.is_file():
        if root.name == "selection.json" or root.name.endswith("_plot_data.csv"):
            return [root]
        return []

    # 同时兼容原始 selection.json 和后处理绘图 CSV，方便复用同一套自动选 k 逻辑。
    paths: List[Path] = []
    direct_selection = root / "selection.json"
    if direct_selection.exists():
        paths.append(direct_selection)
    paths.extend(path for path in root.glob("*/selection.json") if path.is_file())
    paths.extend(path for path in root.glob("*_plot_data.csv") if path.is_file())
    return sorted(set(paths), key=lambda path: str(path))


def _normalize_min_k(min_k: int, num_layers: int) -> int:
    if min_k <= 0:
        raise ValueError("min_k 必须为正数。")
    return min(int(min_k), int(num_layers))


def _resolve_max_k(
    num_layers: int,
    min_k: int,
    max_k: Optional[int],
    max_k_ratio: Optional[float],
) -> int:
    candidates = [int(num_layers)]
    if max_k is not None:
        if max_k <= 0:
            raise ValueError("max_k 必须为正数。")
        candidates.append(int(max_k))
    if max_k_ratio is not None:
        if max_k_ratio <= 0:
            raise ValueError("max_k_ratio 必须为正数。")
        candidates.append(max(1, int(math.ceil(num_layers * float(max_k_ratio)))))
    return max(int(min_k), min(int(num_layers), min(candidates)))


def _rank_scores(
    consensus_scores: Dict[str, Any],
    candidate_layers: Sequence[str],
) -> List[tuple[str, float]]:
    layer_order = {str(layer): index for index, layer in enumerate(candidate_layers)}
    missing_layers = [layer for layer in candidate_layers if layer not in consensus_scores]
    if missing_layers:
        raise ValueError(f"consensus_scores 缺少候选层: {missing_layers}")

    # 排序规则保持和 layer_extraction.selection 一致，分数相同则按候选层顺序稳定打破平局。
    return sorted(
        ((str(layer), float(consensus_scores[str(layer)])) for layer in candidate_layers),
        key=lambda item: (-item[1], layer_order[item[0]]),
    )


def _build_ranked_layers(ranked_scores: Sequence[tuple[str, float]]) -> List[RankedLayerScore]:
    num_layers = len(ranked_scores)
    if num_layers == 0:
        raise ValueError("ranked_scores 不能为空。")

    scores = [score for _, score in ranked_scores]
    max_score = scores[0]
    min_score = scores[-1]
    score_range = max_score - min_score
    ranked_layers: List[RankedLayerScore] = []
    for index, (layer, score) in enumerate(ranked_scores):
        normalized_rank = 0.0 if num_layers == 1 else float(index / (num_layers - 1))
        if abs(score_range) <= 1e-12:
            normalized_score = 1.0
        else:
            normalized_score = float((score - min_score) / score_range)
        baseline = 1.0 - normalized_rank
        next_gap = None
        if index + 1 < num_layers:
            next_gap = float(score - ranked_scores[index + 1][1])
        ranked_layers.append(
            RankedLayerScore(
                rank=index + 1,
                layer=layer,
                score=float(score),
                normalized_rank=normalized_rank,
                normalized_score=normalized_score,
                elbow_distance=float(abs(normalized_score - baseline)),
                next_gap=next_gap,
            )
        )
    return ranked_layers


def _choose_k_by_chord_distance(
    ranked_layers: Sequence[RankedLayerScore],
    min_k: int,
    max_k: int,
) -> int:
    if len(ranked_layers) == 1:
        return 1

    searchable = [
        item
        for item in ranked_layers
        if int(min_k) <= item.rank <= int(max_k)
    ]
    if not searchable:
        return int(min_k)

    # chord elbow：找离首尾直线最远的 rank，适合在分数曲线由陡变平处截断。
    best = max(searchable, key=lambda item: (item.elbow_distance, -item.rank))
    if best.elbow_distance <= 1e-12:
        return int(min_k)
    return int(best.rank)


def _choose_k_by_max_gap(
    ranked_layers: Sequence[RankedLayerScore],
    min_k: int,
    max_k: int,
) -> int:
    searchable = [
        item
        for item in ranked_layers
        if int(min_k) <= item.rank < int(max_k) and item.next_gap is not None
    ]
    if not searchable:
        return int(max_k)

    # max_gap elbow：找相邻共识分下降最大的截断点。
    best = max(searchable, key=lambda item: (float(item.next_gap), -item.rank))
    return int(best.rank)


def select_k_from_selection_payload(
    payload: Dict[str, Any],
    source_path: str | Path = "",
    min_k: int = 1,
    max_k: Optional[int] = None,
    max_k_ratio: Optional[float] = None,
    method: str = "chord",
) -> AutoKResult:
    consensus_scores = payload.get("consensus_scores")
    candidate_layers = payload.get("candidate_layers")
    if not isinstance(consensus_scores, dict):
        raise ValueError("selection.json 缺少 consensus_scores 字段。")
    if not isinstance(candidate_layers, list) or not candidate_layers:
        raise ValueError("selection.json 缺少非空 candidate_layers 字段。")

    ranked_scores = _rank_scores(consensus_scores, [str(layer) for layer in candidate_layers])
    ranked_layers = _build_ranked_layers(ranked_scores)
    resolved_min_k = _normalize_min_k(min_k=min_k, num_layers=len(ranked_layers))
    resolved_max_k = _resolve_max_k(
        num_layers=len(ranked_layers),
        min_k=resolved_min_k,
        max_k=max_k,
        max_k_ratio=max_k_ratio,
    )

    normalized_method = str(method).strip().lower()
    if normalized_method == "chord":
        recommended_k = _choose_k_by_chord_distance(
            ranked_layers=ranked_layers,
            min_k=resolved_min_k,
            max_k=resolved_max_k,
        )
    elif normalized_method == "max_gap":
        recommended_k = _choose_k_by_max_gap(
            ranked_layers=ranked_layers,
            min_k=resolved_min_k,
            max_k=resolved_max_k,
        )
    else:
        raise ValueError(f"不支持的自动选 k 方法: {method}")

    selected_layers = [item.layer for item in ranked_layers[:recommended_k]]
    existing_selected_layers = payload.get("selected_layers") or []
    existing_k = payload.get("k")
    return AutoKResult(
        source_path=str(source_path),
        model=str(payload.get("model", "")),
        dataset=str(payload.get("dataset", "")),
        partition=str(payload.get("partition", "")),
        existing_k=int(existing_k) if existing_k is not None else None,
        recommended_k=int(recommended_k),
        selected_layers=selected_layers,
        existing_selected_layers=[str(layer) for layer in existing_selected_layers],
        method=normalized_method,
        min_k=int(resolved_min_k),
        max_k=int(resolved_max_k),
        max_k_ratio=float(max_k_ratio) if max_k_ratio is not None else None,
        ranked_layers=list(ranked_layers),
    )


def select_k_from_selection_file(
    selection_path: str | Path,
    min_k: int = 1,
    max_k: Optional[int] = None,
    max_k_ratio: Optional[float] = None,
    method: str = "chord",
) -> AutoKResult:
    path = Path(selection_path)
    payload = load_selection_payload(path)
    return select_k_from_selection_payload(
        payload=payload,
        source_path=path,
        min_k=min_k,
        max_k=max_k,
        max_k_ratio=max_k_ratio,
        method=method,
    )


def _truthy(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def load_plot_data_payload(csv_path: str | Path) -> Dict[str, Any]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} 是空 CSV。")

    layer_rows: Dict[str, Dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        layer = str(row.get("layer", "")).strip()
        if not layer or layer in layer_rows:
            continue
        if "consensus_score" not in row:
            raise ValueError(f"{path} 缺少 consensus_score 字段。")
        # plot_data 每个 attack 会重复写同一层的共识分，这里只保留每层第一条记录。
        layer_rows[layer] = {
            "layer": layer,
            "consensus_score": float(row["consensus_score"]),
            "consensus_rank": int(float(row.get("consensus_rank") or row_index + 1)),
            "is_selected": _truthy(row.get("is_selected", "")),
        }

    if not layer_rows:
        raise ValueError(f"{path} 没有可用的层级分数。")

    ordered_items = sorted(
        layer_rows.values(),
        key=lambda item: (int(item["consensus_rank"]), str(item["layer"])),
    )
    selected_items = [item for item in ordered_items if bool(item["is_selected"])]
    first_row = rows[0]
    return {
        "model": str(first_row.get("model_name", "")).strip(),
        "dataset": str(first_row.get("dataset", "")).strip(),
        "partition": str(first_row.get("partition", "")).strip(),
        "k": len(selected_items) if selected_items else None,
        "candidate_layers": [str(item["layer"]) for item in ordered_items],
        "selected_layers": [str(item["layer"]) for item in selected_items],
        "consensus_scores": {
            str(item["layer"]): float(item["consensus_score"])
            for item in ordered_items
        },
    }


def select_k_from_plot_data_file(
    csv_path: str | Path,
    min_k: int = 1,
    max_k: Optional[int] = None,
    max_k_ratio: Optional[float] = None,
    method: str = "chord",
) -> AutoKResult:
    path = Path(csv_path)
    payload = load_plot_data_payload(path)
    return select_k_from_selection_payload(
        payload=payload,
        source_path=path,
        min_k=min_k,
        max_k=max_k,
        max_k_ratio=max_k_ratio,
        method=method,
    )


def select_k_from_score_file(
    score_path: str | Path,
    min_k: int = 1,
    max_k: Optional[int] = None,
    max_k_ratio: Optional[float] = None,
    method: str = "chord",
) -> AutoKResult:
    path = Path(score_path)
    if path.name == "selection.json":
        return select_k_from_selection_file(
            selection_path=path,
            min_k=min_k,
            max_k=max_k,
            max_k_ratio=max_k_ratio,
            method=method,
        )
    if path.name.endswith("_plot_data.csv"):
        return select_k_from_plot_data_file(
            csv_path=path,
            min_k=min_k,
            max_k=max_k,
            max_k_ratio=max_k_ratio,
            method=method,
        )
    raise ValueError(f"不支持的输入文件: {path}")


def write_results_json(results: Sequence[AutoKResult], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([result.to_dict() for result in results], handle, ensure_ascii=False, indent=2)


def write_results_csv(results: Sequence[AutoKResult], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "model",
        "dataset",
        "partition",
        "existing_k",
        "recommended_k",
        "selected_layers",
        "existing_selected_layers",
        "method",
        "min_k",
        "max_k",
        "max_k_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = result.to_dict()
            row["selected_layers"] = ",".join(result.selected_layers)
            row["existing_selected_layers"] = ",".join(result.existing_selected_layers)
            row.pop("ranked_layers", None)
            writer.writerow(row)


def run_select_k(
    selection_paths: Iterable[str | Path],
    min_k: int = 1,
    max_k: Optional[int] = None,
    max_k_ratio: Optional[float] = None,
    method: str = "chord",
) -> List[AutoKResult]:
    return [
        select_k_from_score_file(
            score_path=path,
            min_k=min_k,
            max_k=max_k,
            max_k_ratio=max_k_ratio,
            method=method,
        )
        for path in selection_paths
    ]


def apply_model_k_overrides(
    results: Sequence[AutoKResult],
    model_k_overrides: Dict[str, int],
) -> List[AutoKResult]:
    if not model_k_overrides:
        return list(results)

    normalized_overrides = {
        str(model_name).strip().lower(): int(k)
        for model_name, k in model_k_overrides.items()
    }
    overridden_results: List[AutoKResult] = []
    for result in results:
        model_key = str(result.model).strip().lower()
        if model_key not in normalized_overrides:
            overridden_results.append(result)
            continue

        forced_k = normalized_overrides[model_key]
        if forced_k <= 0:
            raise ValueError(f"{result.model} 的强制 k 必须为正数。")
        resolved_k = min(int(forced_k), len(result.ranked_layers))
        # 用于论文绘图时做人工预算约束；层排序仍来自同一份共识分数曲线。
        overridden_results.append(
            replace(
                result,
                recommended_k=resolved_k,
                selected_layers=[item.layer for item in result.ranked_layers[:resolved_k]],
            )
        )
    return overridden_results


def _slugify(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def _display_model_name(model_name: str) -> str:
    display_names = {
        "lenet5": "LeNet-5",
        "resnet18": "ResNet-18",
        "resnet20": "ResNet-20",
        "resnet34": "ResNet-34",
        "vgg11": "VGG-11",
    }
    return display_names.get(str(model_name).strip().lower(), str(model_name))


def _sort_results_for_figure(results: Sequence[AutoKResult]) -> List[AutoKResult]:
    model_order = {
        "lenet5": 0,
        "resnet20": 1,
        "resnet18": 2,
        "resnet34": 3,
        "vgg11": 4,
    }
    return sorted(
        results,
        key=lambda result: (
            model_order.get(str(result.model).strip().lower(), 999),
            str(result.model),
            str(result.dataset),
        ),
    )


def plot_auto_k_result(result: AutoKResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ranks = [item.rank for item in result.ranked_layers]
    scores = [item.score for item in result.ranked_layers]
    normalized_scores = [item.normalized_score for item in result.ranked_layers]
    baseline = [1.0 - item.normalized_rank for item in result.ranked_layers]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(ranks, scores, marker="o", linewidth=1.8, color="#2b6cb0")
    axes[0].axvline(result.recommended_k, linestyle="--", linewidth=1.4, color="#c53030")
    axes[0].scatter(
        ranks[: result.recommended_k],
        scores[: result.recommended_k],
        color="#dd6b20",
        zorder=3,
        label=f"auto k={result.recommended_k}",
    )
    axes[0].set_title(f"{result.model} consensus scores")
    axes[0].set_xlabel("rank")
    axes[0].set_ylabel("consensus score")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(ranks, normalized_scores, marker="o", linewidth=1.8, label="normalized score")
    axes[1].plot(ranks, baseline, linestyle="--", linewidth=1.4, label="chord baseline")
    elbow = result.ranked_layers[result.recommended_k - 1]
    axes[1].scatter(
        [elbow.rank],
        [elbow.normalized_score],
        color="#c53030",
        zorder=4,
        label="selected elbow",
    )
    axes[1].axvline(result.recommended_k, linestyle="--", linewidth=1.2, color="#c53030")
    axes[1].set_xlabel("rank")
    axes[1].set_ylabel("normalized value")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    tick_step = max(1, int(math.ceil(len(ranks) / 18)))
    axes[0].set_xticks(ranks[::tick_step])
    axes[1].set_xticks(ranks[::tick_step])

    fig.suptitle(
        f"auto-k by {result.method}: {result.model} / {result.dataset} / k={result.recommended_k}",
        fontsize=13,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_result_plots(
    results: Sequence[AutoKResult],
    output_dir: str | Path,
    image_format: str = "png",
) -> List[Path]:
    output_root = Path(output_dir)
    written_paths: List[Path] = []
    for result in results:
        filename = (
            f"{_slugify(result.model)}_{_slugify(result.dataset)}_"
            f"{_slugify(result.method)}_k{result.recommended_k}.{image_format}"
        )
        written_paths.append(plot_auto_k_result(result, output_root / filename))
    return written_paths


def plot_combined_auto_k_results(
    results: Sequence[AutoKResult],
    output_path: str | Path,
    title: str = "",
) -> Path:
    if not results:
        raise ValueError("results 不能为空。")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_results = _sort_results_for_figure(results)

    style = {
        "font.family": "Liberation Serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "Times"],
        "font.size": 14.5,
        "axes.titlesize": 15.5,
        "axes.labelsize": 14.5,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(12.4, 7.0),
            sharey=True,
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.095,
            right=0.985,
            bottom=0.135,
            top=0.965,
            wspace=0.24,
            hspace=0.34,
        )
        flat_axes = list(axes.ravel())
        letters = ["a", "b", "c", "d", "e"]

        for index, result in enumerate(ordered_results[:5]):
            axis = flat_axes[index]
            ranks = [item.rank for item in result.ranked_layers]
            normalized_scores = [item.normalized_score for item in result.ranked_layers]
            baseline = [1.0 - item.normalized_rank for item in result.ranked_layers]
            elbow = result.ranked_layers[result.recommended_k - 1]

            # 用浅色区域标出最终进入关键层集合的 rank，图中不放长层名以保证论文版可读性。
            axis.axvspan(
                0.5,
                result.recommended_k + 0.5,
                color="#f8e9a1",
                alpha=0.45,
                linewidth=0,
                zorder=0,
            )
            axis.plot(
                ranks,
                baseline,
                color="#9ca3af",
                linewidth=1.15,
                linestyle=(0, (4, 2)),
                zorder=1,
            )
            axis.plot(
                ranks,
                normalized_scores,
                color="#1f4e79",
                linewidth=1.75,
                marker="o",
                markersize=3.4,
                markerfacecolor="white",
                markeredgewidth=0.9,
                zorder=2,
            )
            axis.axvline(
                result.recommended_k,
                color="#b91c1c",
                linewidth=1.25,
                linestyle="--",
                zorder=3,
            )
            axis.scatter(
                [elbow.rank],
                [elbow.normalized_score],
                s=34,
                color="#b91c1c",
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            axis.text(
                0.98,
                0.10,
                f"k = {result.recommended_k}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=14.5,
                color="#7f1d1d",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "#e5e7eb",
                    "linewidth": 0.6,
                    "alpha": 0.92,
                },
            )
            axis.set_title(
                f"({letters[index]}) {_display_model_name(result.model)}",
                loc="left",
                pad=4,
                fontweight="semibold",
            )
            axis.set_xlim(0.5, max(ranks) + 0.5)
            axis.set_ylim(-0.04, 1.04)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            axis.grid(axis="y", color="#d1d5db", alpha=0.55, linewidth=0.65)
            axis.grid(axis="x", color="#e5e7eb", alpha=0.25, linewidth=0.5)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        for axis in flat_axes[len(ordered_results[:5]):5]:
            axis.axis("off")

        legend_axis = flat_axes[5]
        legend_axis.axis("off")
        legend_handles = [
            Line2D(
                [0],
                [0],
                color="#1f4e79",
                marker="o",
                markerfacecolor="white",
                linewidth=1.75,
                markersize=4,
                label="Normalized consensus score",
            ),
            Line2D(
                [0],
                [0],
                color="#9ca3af",
                linestyle=(0, (4, 2)),
                linewidth=1.15,
                label="Chord baseline",
            ),
            Line2D(
                [0],
                [0],
                color="#b91c1c",
                linestyle="--",
                linewidth=1.25,
                label="Selected elbow",
            ),
        ]
        legend_axis.legend(
            handles=legend_handles,
            loc="upper left",
            frameon=False,
            borderaxespad=0.0,
        )
        summary_lines = [
            f"{_display_model_name(result.model)}: k={result.recommended_k}"
            for result in ordered_results[:5]
        ]
        legend_axis.text(
            0.0,
            0.56,
            "Selected budgets\n" + "\n".join(summary_lines),
            ha="left",
            va="top",
            fontsize=13.3,
            linespacing=1.55,
        )

        fig.supxlabel("Layer rank sorted by consensus score", y=0.055, fontsize=15)
        fig.supylabel("Normalized consensus score", x=0.035, fontsize=15)
        if title:
            fig.suptitle(title, y=1.01, fontsize=15.5, fontweight="semibold")
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)

    return path
