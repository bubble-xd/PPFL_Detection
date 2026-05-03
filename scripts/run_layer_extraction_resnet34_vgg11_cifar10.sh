#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PARTITION="${PARTITION:-iid}"

echo "[Run] task=layer_extraction models=resnet34,vgg11 partition=${PARTITION}"

# 通过独立的 Python 进程临时覆盖分区配置，避免改动仓库里的默认 Config。
"${PYTHON_BIN}" - "${PARTITION}" <<'PY'
import sys

from config import Config
from layer_extraction import LayerExtractionSettings, run_layer_extraction

partition = sys.argv[1]

# layer extraction 只需要当前批次的划分方式，其他训练超参数继续沿用 config.py。
Config.PARTITION = partition

# 批量模型模式下，lenet5 固定映射到 MNIST，其余模型固定映射到 CIFAR10；
# 因此这里只需要传模型列表，就能得到 resnet34+cifar10 与 vgg11+cifar10 的提取结果。
settings = LayerExtractionSettings.from_config(
    base_config=Config,
    selected_models=["resnet34", "vgg11"],
)

result = run_layer_extraction(settings)

if "runs" in result:
    for run_result in result["runs"]:
        print(
            f"[Done] model={run_result['model']} "
            f"dataset={run_result['dataset']} "
            f"partition={run_result['partition']} "
            f"results={run_result['output_dir']}"
        )
else:
    print(
        f"[Done] model={result['model']} "
        f"dataset={result['dataset']} "
        f"partition={result['partition']} "
        f"results={result['output_dir']}"
    )
PY
