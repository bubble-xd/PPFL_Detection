#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PARTITION="${PARTITION:-iid}"

run_one() {
    local model="$1"
    local dataset="$2"

    echo "[Run] model=${model} dataset=${dataset} partition=${PARTITION}"

    # 通过独立的 Python 进程临时覆盖 Config，避免直接改仓库里的默认配置。
    "${PYTHON_BIN}" - "${model}" "${dataset}" "${PARTITION}" <<'PY'
import sys

from config import Config
from utils.benchmark import run_all_experiments

model, dataset, partition = sys.argv[1:4]

# 这里只改当前这次运行的实验组合，其余攻击、方法、超参数继续沿用 config.py。
Config.MODEL = model
Config.DATASET = dataset
Config.PARTITION = partition

output_dir = run_all_experiments(Config)
print(f"[Done] model={model} dataset={dataset} partition={partition} results={output_dir}")
PY
}

# 这三个模型当前都走 CIFAR10 实验配置，便于直接批量启动对比实验。
run_one "resnet18" "cifar10"
run_one "resnet34" "cifar10"
run_one "vgg11" "cifar10"
