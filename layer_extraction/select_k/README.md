# Select K

这个目录用于**读取已有 layer extraction 结果并自动估计选层数量 `k`**。

它只读取已有分数文件，不会重新训练模型，也不会重新执行攻击模拟。支持两类输入：

- `selection.json`：使用其中的 `candidate_layers` 和 `consensus_scores`
- `*_plot_data.csv`：使用其中的 `layer / consensus_score / consensus_rank`，同一层的重复 attack 行会自动去重

## 默认用法

```bash
python -m layer_extraction.select_k
```

默认扫描：

```text
results/layer_extraction/*/selection.json
results/layer_extraction/*_plot_data.csv
```

也可以直接读取当前目录下准备好的 plot data：

```bash
python -m layer_extraction.select_k layer_extraction/select_k/layer_extraction_data
```

## 方法

默认方法是 `chord`：

1. 按 `consensus_scores` 从高到低排序。
2. 将 rank 和 score 归一化到 `[0, 1]`。
3. 连接首尾点，选择距离这条直线最远的 rank 作为 elbow。
4. 将 elbow rank 作为自动 `k`。

也可以使用更直接的相邻分数最大下降：

```bash
python -m layer_extraction.select_k --method max_gap
```

## 限制搜索范围

如果担心自动结果选太多层，可以加边界：

```bash
python -m layer_extraction.select_k --min-k 2 --max-k 5
```

或者按候选层比例限制：

```bash
python -m layer_extraction.select_k --min-k 2 --max-k-ratio 0.2
```

## 导出结果

```bash
python -m layer_extraction.select_k \
  --output-json results/layer_extraction/select_k_summary.json \
  --output-csv results/layer_extraction/select_k_summary.csv
```

## 可视化

```bash
python -m layer_extraction.select_k layer_extraction/select_k/layer_extraction_data \
  --plot-dir layer_extraction/select_k/layer_extraction_data/select_k_plots
```

每个模型会输出一张图：

- 上半部分：原始 `consensus_score` 随 rank 的下降曲线
- 下半部分：归一化分数、首尾连线基准和自动 elbow 位置

如果要导出五个模型合在一起的论文版 PDF：

```bash
python -m layer_extraction.select_k layer_extraction/select_k/layer_extraction_data \
  --combined-pdf layer_extraction/select_k/layer_extraction_data/select_k_elbow_combined.pdf
```

如果论文图中需要对某个小模型设置人工预算约束，例如 LeNet-5 只取 1 层：

```bash
python -m layer_extraction.select_k layer_extraction/select_k/layer_extraction_data \
  --force-k lenet5=1 \
  --combined-pdf layer_extraction/select_k/layer_extraction_data/select_k_elbow_combined.pdf
```
