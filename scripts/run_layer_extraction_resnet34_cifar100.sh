#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PARTITION="${PARTITION:-iid}"

echo "[Run] task=layer_extraction model=resnet34 dataset=cifar100 partition=${PARTITION}"

"${PYTHON_BIN}" - "${PARTITION}" <<'PY'
import sys

from config import Config
from layer_extraction import LayerExtractionSettings, run_layer_extraction

partition = sys.argv[1]

# 只固定本次 layer_extraction 的实验对象；
# 攻击列表、攻击强度和 CIFAR100 覆盖项继续统一读取 config.py。
Config.MODEL = "resnet34"
Config.DATASET = "cifar100"
Config.PARTITION = partition

settings = LayerExtractionSettings.from_config(base_config=Config)
result = run_layer_extraction(settings)

print(
    f"[Done] model={result['model']} "
    f"dataset={result['dataset']} "
    f"partition={result['partition']} "
    f"results={result['output_dir']}"
)
print(f"[Done] selected_layers={result['selected_layers']}")
PY
