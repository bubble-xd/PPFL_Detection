from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors


def resolve_cosine_heatmap_rounds(
    num_rounds: int,
    configured_rounds: Sequence[int] | None = None,
) -> List[int]:
    if configured_rounds:
        rounds = sorted(
            {
                int(round_idx)
                for round_idx in configured_rounds
                if 1 <= int(round_idx) <= int(num_rounds)
            }
        )
        if rounds:
            return rounds
    mid_round = max(1, int(num_rounds) // 2)
    return sorted({1, mid_round, int(num_rounds)})


def build_fixed_client_order(
    num_clients: int,
    malicious_ids: Sequence[int],
) -> List[int]:
    malicious_sorted = sorted({int(client_id) for client_id in malicious_ids})
    malicious_set = set(malicious_sorted)
    benign_sorted = [client_id for client_id in range(int(num_clients)) if client_id not in malicious_set]
    return benign_sorted + malicious_sorted


def pairwise_cosine_similarity(
    feature_matrix: torch.Tensor,
    eps: float = 1e-12,
) -> np.ndarray:
    matrix = feature_matrix.detach().to(dtype=torch.float32)
    if matrix.dim() != 2:
        raise ValueError("feature_matrix must be a 2D tensor.")
    # 特征矩阵如果已经在 CUDA 上，就直接在 GPU 完成归一化和矩阵乘；
    # 最终只把 20x20 级别的相似度矩阵搬回 CPU 供 NumPy/绘图使用。
    norms = torch.norm(matrix, p=2, dim=1, keepdim=True).clamp_min(float(eps))
    normalized = matrix / norms
    similarities = normalized @ normalized.t()
    return similarities.clamp(min=-1.0, max=1.0).cpu().numpy()


def _safe_mean(values: np.ndarray) -> float:
    """返回均值；当输入为空时返回 NaN，避免伪造 0 值。"""
    return float(np.mean(values)) if values.size > 0 else float("nan")


def _safe_mean_finite(values: np.ndarray) -> float:
    """只对有限值求均值，避免单个 NaN 把整个统计量污染掉。"""
    finite_values = np.asarray(values, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    return float(np.mean(finite_values)) if finite_values.size > 0 else float("nan")


def _extract_block_values(
    similarity_matrix: np.ndarray,
    row_indices: Sequence[int],
    col_indices: Sequence[int],
    exclude_diagonal: bool = False,
) -> np.ndarray:
    """从相似度矩阵中抽取一个子块，并按需去掉对角线元素。"""
    if not row_indices or not col_indices:
        return np.asarray([], dtype=np.float32)

    row_array = np.asarray([int(index) for index in row_indices], dtype=np.int64)
    col_array = np.asarray([int(index) for index in col_indices], dtype=np.int64)
    block = np.asarray(similarity_matrix[np.ix_(row_array, col_array)], dtype=np.float32)

    if exclude_diagonal:
        # 同组统计时只保留上三角，避免 (i, j) 和 (j, i) 被重复计数。
        if row_array.shape == col_array.shape and np.array_equal(row_array, col_array):
            block = block[np.triu_indices(block.shape[0], k=1)]
        else:
            same_client_mask = row_array[:, None] == col_array[None, :]
            block = block[~same_client_mask]
    else:
        block = block.reshape(-1)

    return np.asarray(block, dtype=np.float32)


def _compute_silhouette_from_cosine_similarity(
    similarity_matrix: np.ndarray,
    benign_positions: Sequence[int],
    malicious_positions: Sequence[int],
) -> Dict[str, float]:
    """
    基于余弦相似度矩阵计算 silhouette score。

    计算步骤与用户给出的定义一致：
    1. 先把余弦相似度转成距离：d(i, j) = 1 - cos(i, j)
    2. 再按真实标签划分 benign / malicious
    3. 对每个客户端计算：
       - a(i): 到同类其他客户端的平均距离
       - b(i): 到异类客户端的平均距离
       - s(i) = (b(i) - a(i)) / max(a(i), b(i))

    特殊情况说明：
    - 如果某个客户端所在类别只有它自己一个样本，则该客户端 silhouette 记为 0。
      这是常见实现对 singleton cluster 的处理方式。
    - 如果压根没有异类客户端，则 silhouette 不可定义，返回 NaN。
    """
    distance_matrix = np.clip(1.0 - np.asarray(similarity_matrix, dtype=np.float32), 0.0, 2.0)
    num_clients = int(distance_matrix.shape[0])
    silhouette_values = np.full(num_clients, np.nan, dtype=np.float32)

    benign_set = {int(position) for position in benign_positions}
    malicious_set = {int(position) for position in malicious_positions}
    singleton_client_count = 0

    for client_index in range(num_clients):
        if client_index in benign_set:
            same_positions = [int(position) for position in benign_positions if int(position) != client_index]
            other_positions = [int(position) for position in malicious_positions]
        elif client_index in malicious_set:
            same_positions = [int(position) for position in malicious_positions if int(position) != client_index]
            other_positions = [int(position) for position in benign_positions]
        else:
            continue

        # 没有异类时 silhouette 无法定义，这里保持 NaN。
        if not other_positions:
            continue

        # 同类只有自己一个样本时，按常见约定将 silhouette 置为 0。
        if not same_positions:
            silhouette_values[client_index] = 0.0
            singleton_client_count += 1
            continue

        intra_class_distance = float(np.mean(distance_matrix[client_index, same_positions]))
        inter_class_distance = float(np.mean(distance_matrix[client_index, other_positions]))
        denominator = max(intra_class_distance, inter_class_distance)
        if denominator <= 1e-12:
            silhouette_values[client_index] = 0.0
            continue
        silhouette_values[client_index] = (
            inter_class_distance - intra_class_distance
        ) / denominator

    benign_silhouette_values = silhouette_values[np.asarray(list(benign_positions), dtype=np.int64)]
    malicious_silhouette_values = silhouette_values[np.asarray(list(malicious_positions), dtype=np.int64)]

    return {
        "silhouette_score": _safe_mean_finite(silhouette_values),
        "mean_benign_silhouette": _safe_mean_finite(benign_silhouette_values),
        "mean_malicious_silhouette": _safe_mean_finite(malicious_silhouette_values),
        "num_valid_silhouette_clients": float(np.isfinite(silhouette_values).sum()),
        "num_singleton_silhouette_clients": float(singleton_client_count),
    }


def compute_cosine_group_metrics(
    similarity_matrix: np.ndarray,
    client_order: Sequence[int],
    malicious_ids: Sequence[int],
) -> Dict[str, float]:
    """
    基于余弦相似度矩阵计算分组统计量。

    这里的主指标 BM-Gap 定义为：
        BM-Gap = mean(cos_ben-ben) - mean(cos_ben-mal)
    数值越大，表示良性客户端内部越紧、良恶之间越远，可分性越强。
    """
    matrix = np.asarray(similarity_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarity_matrix must be a square 2D matrix.")
    if matrix.shape[0] != len(client_order):
        raise ValueError("client_order length must match similarity_matrix size.")

    malicious_set = {int(client_id) for client_id in malicious_ids}
    benign_positions = [
        position
        for position, client_id in enumerate(client_order)
        if int(client_id) not in malicious_set
    ]
    malicious_positions = [
        position
        for position, client_id in enumerate(client_order)
        if int(client_id) in malicious_set
    ]

    benign_benign_values = _extract_block_values(
        similarity_matrix=matrix,
        row_indices=benign_positions,
        col_indices=benign_positions,
        exclude_diagonal=True,
    )
    benign_malicious_values = _extract_block_values(
        similarity_matrix=matrix,
        row_indices=benign_positions,
        col_indices=malicious_positions,
    )
    malicious_malicious_values = _extract_block_values(
        similarity_matrix=matrix,
        row_indices=malicious_positions,
        col_indices=malicious_positions,
        exclude_diagonal=True,
    )
    silhouette_metrics = _compute_silhouette_from_cosine_similarity(
        similarity_matrix=matrix,
        benign_positions=benign_positions,
        malicious_positions=malicious_positions,
    )

    mean_benign_benign_cosine = _safe_mean(benign_benign_values)
    mean_benign_malicious_cosine = _safe_mean(benign_malicious_values)
    mean_malicious_malicious_cosine = _safe_mean(malicious_malicious_values)
    bm_gap = (
        float(mean_benign_benign_cosine - mean_benign_malicious_cosine)
        if not np.isnan(mean_benign_benign_cosine) and not np.isnan(mean_benign_malicious_cosine)
        else float("nan")
    )

    return {
        "bm_gap": bm_gap,
        "mean_benign_benign_cosine": mean_benign_benign_cosine,
        "mean_benign_malicious_cosine": mean_benign_malicious_cosine,
        "mean_malicious_malicious_cosine": mean_malicious_malicious_cosine,
        "num_benign_clients": float(len(benign_positions)),
        "num_malicious_clients": float(len(malicious_positions)),
        "num_benign_benign_pairs": float(benign_benign_values.size),
        "num_benign_malicious_pairs": float(benign_malicious_values.size),
        "num_malicious_malicious_pairs": float(malicious_malicious_values.size),
        "silhouette_score": silhouette_metrics["silhouette_score"],
        "mean_benign_silhouette": silhouette_metrics["mean_benign_silhouette"],
        "mean_malicious_silhouette": silhouette_metrics["mean_malicious_silhouette"],
        "num_valid_silhouette_clients": silhouette_metrics["num_valid_silhouette_clients"],
        "num_singleton_silhouette_clients": silhouette_metrics["num_singleton_silhouette_clients"],
    }


def _build_client_labels(
    client_order: Sequence[int],
    malicious_ids: Sequence[int],
) -> List[str]:
    malicious_set = {int(client_id) for client_id in malicious_ids}
    return [
        f"C{int(client_id):02d}" + ("*" if int(client_id) in malicious_set else "")
        for client_id in client_order
    ]


def _write_heatmap_data_csv(
    csv_path: str,
    similarity_matrices: Dict[str, np.ndarray],
    feature_metrics: Dict[str, Dict[str, float]],
    labels: Sequence[str],
    client_order: Sequence[int],
    malicious_ids: Sequence[int],
    feature_display_names: Dict[str, str],
    similarity_space_tag: str,
    similarity_space_description: str,
) -> None:
    malicious_set = {int(client_id) for client_id in malicious_ids}
    fieldnames = [
        "similarity_space_tag",
        "similarity_space_description",
        "feature_mode",
        "feature_display_name",
        "bm_gap",
        "mean_benign_benign_cosine",
        "mean_benign_malicious_cosine",
        "mean_malicious_malicious_cosine",
        "silhouette_score",
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
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        # 一个 CSV 同时把重画热力图需要的矩阵值和标题统计量带齐，
        # 后续无论是重排客户端、改配色还是单独画某个 feature，都不必再翻别的文件。
        for feature_mode, similarity_matrix in similarity_matrices.items():
            metrics = feature_metrics[feature_mode]
            display_name = str(feature_display_names.get(feature_mode, feature_mode))
            for row_position, row_client_id in enumerate(client_order):
                for col_position, col_client_id in enumerate(client_order):
                    writer.writerow(
                        {
                            "similarity_space_tag": str(similarity_space_tag),
                            "similarity_space_description": str(similarity_space_description),
                            "feature_mode": str(feature_mode),
                            "feature_display_name": display_name,
                            "bm_gap": f"{float(metrics['bm_gap']):.6f}",
                            "mean_benign_benign_cosine": f"{float(metrics['mean_benign_benign_cosine']):.6f}",
                            "mean_benign_malicious_cosine": f"{float(metrics['mean_benign_malicious_cosine']):.6f}",
                            "mean_malicious_malicious_cosine": f"{float(metrics['mean_malicious_malicious_cosine']):.6f}",
                            "silhouette_score": f"{float(metrics['silhouette_score']):.6f}",
                            "row_position": int(row_position),
                            "col_position": int(col_position),
                            "row_client_id": int(row_client_id),
                            "col_client_id": int(col_client_id),
                            "row_label": str(labels[row_position]),
                            "col_label": str(labels[col_position]),
                            "row_is_malicious": int(int(row_client_id) in malicious_set),
                            "col_is_malicious": int(int(col_client_id) in malicious_set),
                            "cosine_similarity": f"{float(similarity_matrix[row_position, col_position]):.6f}",
                        }
                    )


def _resolve_color_limits(similarity_matrices: Sequence[np.ndarray]) -> tuple[float, float]:
    off_diagonal_values: List[np.ndarray] = []
    for matrix in similarity_matrices:
        if matrix.shape[0] <= 1:
            continue
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        values = matrix[mask]
        if values.size > 0:
            off_diagonal_values.append(values)

    if not off_diagonal_values:
        return 0.0, 1.0

    values = np.concatenate(off_diagonal_values, axis=0)
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if abs(value_max - value_min) < 1e-6:
        padding = 0.05
        return max(-1.0, value_min - padding), min(1.0, value_max + padding)

    padding = 0.05 * (value_max - value_min)
    color_min = max(-1.0, value_min - padding)
    color_max = 1.0
    if color_max - color_min < 1e-6:
        color_min = max(-1.0, color_max - 0.1)
    return color_min, color_max


def _annotation_text_style(
    value: float,
    cmap,
    norm,
) -> str:
    rgba = cmap(norm(value))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "black" if luminance > 0.58 else "white"


def _annotation_fontsize(matrix_size: int) -> float:
    if matrix_size <= 10:
        return 7.0
    if matrix_size <= 15:
        return 5.8
    return 4.6


def _resolve_heatmap_grid(feature_count: int) -> tuple[int, int]:
    if int(feature_count) <= 0:
        raise ValueError("feature_count must be positive.")
    if int(feature_count) == 1:
        return 1, 1

    # 每行最多放三个热力图，保证常见的 3/5 个特征模式都能更紧凑地横向对比。
    num_cols = min(3, int(feature_count))
    num_rows = (int(feature_count) + num_cols - 1) // num_cols
    return num_rows, num_cols


def save_cosine_heatmaps(
    output_dir: str,
    experiment_name: str,
    attack_name: str,
    attack_mode: str,
    round_idx: int,
    feature_matrices: Dict[str, torch.Tensor],
    client_order: Sequence[int],
    malicious_ids: Sequence[int],
    feature_display_names: Dict[str, str],
    artifact_subdir: str = "default",
    similarity_space_tag: str = "default",
    similarity_space_description: str = "",
) -> List[str]:
    attack_dir = os.path.join(output_dir, "cosine_heatmaps", str(artifact_subdir), f"{attack_name}_{attack_mode}")
    os.makedirs(attack_dir, exist_ok=True)

    labels = _build_client_labels(client_order=client_order, malicious_ids=malicious_ids)
    malicious_count = len(malicious_ids)
    split_index = len(client_order) - malicious_count
    ordered_indices = np.asarray(client_order, dtype=np.int64)
    ordered_similarity_matrices = {
        feature_mode: pairwise_cosine_similarity(feature_matrix)[ordered_indices][:, ordered_indices]
        for feature_mode, feature_matrix in feature_matrices.items()
    }
    feature_metrics = {
        feature_mode: compute_cosine_group_metrics(
            similarity_matrix=similarity_matrix,
            client_order=client_order,
            malicious_ids=malicious_ids,
        )
        for feature_mode, similarity_matrix in ordered_similarity_matrices.items()
    }
    color_min, color_max = _resolve_color_limits(list(ordered_similarity_matrices.values()))
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=color_min, vmax=color_max)

    num_rows, num_cols = _resolve_heatmap_grid(len(ordered_similarity_matrices))
    figure = plt.figure(
        figsize=(7.0 * num_cols + 1.3, 5.5 * num_rows + 2.0)
    )
    grid = figure.add_gridspec(
        num_rows,
        num_cols + 1,
        width_ratios=[*([1.0] * num_cols), 0.06],
        height_ratios=[*([1.0] * num_rows)],
        wspace=0.28,
        hspace=0.28,
    )
    axes_flat = [
        figure.add_subplot(grid[row_idx, col_idx])
        for row_idx in range(num_rows)
        for col_idx in range(num_cols)
    ]
    colorbar_axis = figure.add_subplot(grid[:, num_cols])
    image = None
    saved_artifact_paths: List[str] = []
    annotation_fontsize = _annotation_fontsize(len(labels))
    data_csv_path = os.path.join(attack_dir, f"round_{int(round_idx):03d}_heatmap_data.csv")
    _write_heatmap_data_csv(
        csv_path=data_csv_path,
        similarity_matrices=ordered_similarity_matrices,
        feature_metrics=feature_metrics,
        labels=labels,
        client_order=client_order,
        malicious_ids=malicious_ids,
        feature_display_names=feature_display_names,
        similarity_space_tag=similarity_space_tag,
        similarity_space_description=similarity_space_description,
    )
    saved_artifact_paths.append(data_csv_path)

    for axis, (feature_mode, similarity_matrix) in zip(axes_flat, ordered_similarity_matrices.items()):
        display_name = str(feature_display_names.get(feature_mode, feature_mode))
        title = display_name if display_name.isascii() else str(feature_mode)
        image = axis.imshow(similarity_matrix, norm=norm, cmap=cmap)
        axis.set_title(title)
        axis.set_xticks(range(len(labels)))
        axis.set_yticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=90, fontsize=9)
        axis.set_yticklabels(labels, fontsize=9)

        if 0 < split_index < len(client_order):
            axis.axhline(split_index - 0.5, color="black", linewidth=1.2)
            axis.axvline(split_index - 0.5, color="black", linewidth=1.2)

        axis.set_xticks(np.arange(-0.5, len(labels), 1.0), minor=True)
        axis.set_yticks(np.arange(-0.5, len(labels), 1.0), minor=True)
        axis.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        axis.tick_params(which="minor", bottom=False, left=False)

        malicious_set = {int(client_id) for client_id in malicious_ids}
        for tick_label, client_id in zip(axis.get_xticklabels(), client_order):
            if int(client_id) in malicious_set:
                tick_label.set_color("firebrick")
                tick_label.set_fontweight("bold")
        for tick_label, client_id in zip(axis.get_yticklabels(), client_order):
            if int(client_id) in malicious_set:
                tick_label.set_color("firebrick")
                tick_label.set_fontweight("bold")

        for row_idx in range(similarity_matrix.shape[0]):
            for col_idx in range(similarity_matrix.shape[1]):
                value = float(similarity_matrix[row_idx, col_idx])
                axis.text(
                    col_idx,
                    row_idx,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=annotation_fontsize,
                    color=_annotation_text_style(value, cmap=cmap, norm=norm),
                )

    for axis in axes_flat[len(feature_matrices) :]:
        axis.axis("off")

    if image is not None:
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        tick_count = 6
        colorbar.set_ticks(np.linspace(color_min, color_max, tick_count))
        colorbar.ax.set_ylabel("Cosine similarity", rotation=270, labelpad=14)

    title_lines = [
        f"{experiment_name} | attack={attack_name} | mode={attack_mode} | round={int(round_idx)}"
    ]
    if str(similarity_space_description).strip():
        title_lines.append(str(similarity_space_description).strip())
    figure.suptitle("\n".join(title_lines), fontsize=14)
    figure.text(
        0.5,
        0.03,
        (
            "True malicious clients are fixed at the end and marked with *. "
            "Larger BM-Gap means benign clients are tighter while benign-malicious pairs are farther apart."
        ),
        ha="center",
        fontsize=10,
    )
    figure.subplots_adjust(left=0.07, right=0.92, bottom=0.08, top=0.92)

    figure_path = os.path.join(attack_dir, f"round_{int(round_idx):03d}_cosine_heatmaps.png")
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return [figure_path, *saved_artifact_paths]
