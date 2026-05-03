from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from .elbow import (
    AutoKResult,
    apply_model_k_overrides,
    find_score_paths,
    plot_combined_auto_k_results,
    run_select_k,
    write_result_plots,
    write_results_csv,
    write_results_json,
)


def _resolve_input_paths(paths: Iterable[str], results_root: str) -> List[Path]:
    explicit_paths = [Path(path) for path in paths]
    if not explicit_paths:
        return find_score_paths(results_root)

    score_paths: List[Path] = []
    for path in explicit_paths:
        if path.is_file():
            if path.name != "selection.json" and not path.name.endswith("_plot_data.csv"):
                raise ValueError(f"输入文件必须是 selection.json 或 *_plot_data.csv: {path}")
            score_paths.append(path)
            continue
        if path.is_dir():
            score_paths.extend(find_score_paths(path))
            continue
        raise FileNotFoundError(f"输入路径不存在: {path}")
    return sorted(set(score_paths), key=lambda item: str(item))


def _format_layers(layers: List[str]) -> str:
    return "[" + ", ".join(layers) + "]"


def _parse_force_k(values: List[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--force-k 必须使用 MODEL=K 格式: {value}")
        model_name, k_text = value.split("=", 1)
        model_name = model_name.strip().lower()
        if not model_name:
            raise ValueError(f"--force-k 缺少模型名: {value}")
        overrides[model_name] = int(k_text)
    return overrides


def _print_results(results: List[AutoKResult]) -> None:
    if not results:
        print("没有找到 selection.json 或 *_plot_data.csv。")
        return

    for result in results:
        print(
            " | ".join(
                [
                    f"model={result.model}",
                    f"dataset={result.dataset}",
                    f"partition={result.partition}",
                    f"old_k={result.existing_k}",
                    f"auto_k={result.recommended_k}",
                    f"layers={_format_layers(result.selected_layers)}",
                    f"source={result.source_path}",
                ]
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从已有 layer_extraction/selection.json 中用肘部法则自动估计 k。"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="可选的 selection.json / *_plot_data.csv 文件或包含这些文件的目录；不传则扫描 --results-root。",
    )
    parser.add_argument(
        "--results-root",
        default="results/layer_extraction",
        help="默认扫描的 layer_extraction 结果目录。",
    )
    parser.add_argument(
        "--method",
        choices=["chord", "max_gap"],
        default="chord",
        help="chord 使用首尾连线最大距离；max_gap 使用相邻分数最大下降。",
    )
    parser.add_argument("--min-k", type=int, default=1, help="自动搜索的最小 k。")
    parser.add_argument("--max-k", type=int, default=None, help="自动搜索的最大 k。")
    parser.add_argument(
        "--max-k-ratio",
        type=float,
        default=None,
        help="按候选层数量限制最大 k，例如 0.2 表示最多取 20%% 候选层。",
    )
    parser.add_argument("--output-json", default=None, help="可选：导出完整自动选 k 结果 JSON。")
    parser.add_argument("--output-csv", default=None, help="可选：导出自动选 k 摘要 CSV。")
    parser.add_argument("--plot-dir", default=None, help="可选：把每个模型的肘部曲线图保存到该目录。")
    parser.add_argument("--plot-format", default="png", help="可选：图像格式，默认 png。")
    parser.add_argument("--combined-pdf", default=None, help="可选：导出五个模型合在一起的论文版 PDF。")
    parser.add_argument(
        "--combined-title",
        default="",
        help="可选：合成 PDF 的英文标题。",
    )
    parser.add_argument(
        "--force-k",
        action="append",
        default=[],
        help="可选：对指定模型强制使用某个 k，例如 --force-k lenet5=1。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    score_paths = _resolve_input_paths(paths=args.paths, results_root=args.results_root)
    results = run_select_k(
        selection_paths=score_paths,
        min_k=args.min_k,
        max_k=args.max_k,
        max_k_ratio=args.max_k_ratio,
        method=args.method,
    )
    results = apply_model_k_overrides(
        results=results,
        model_k_overrides=_parse_force_k(args.force_k),
    )
    _print_results(results)

    if args.output_json:
        write_results_json(results=results, output_path=args.output_json)
        print(f"wrote_json={args.output_json}")
    if args.output_csv:
        write_results_csv(results=results, output_path=args.output_csv)
        print(f"wrote_csv={args.output_csv}")
    if args.plot_dir:
        plot_paths = write_result_plots(
            results=results,
            output_dir=args.plot_dir,
            image_format=args.plot_format,
        )
        for plot_path in plot_paths:
            print(f"wrote_plot={plot_path}")
    if args.combined_pdf:
        combined_path = plot_combined_auto_k_results(
            results=results,
            output_path=args.combined_pdf,
            title=args.combined_title,
        )
        print(f"wrote_combined_pdf={combined_path}")


if __name__ == "__main__":
    main()
