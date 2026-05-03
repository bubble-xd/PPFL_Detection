# Layer Extraction

## 目标

这个目录实现一个**离线关键层提取工具**，用于在正式运行 `selected_layers` 特征模式之前，先根据服务器侧的纯净代理数据集和已知攻击集合，挑出对攻击更敏感的层前缀。

它不会自动改写主流程配置，而是导出 `selection.json` 和 `layer_scores.csv`，你可以把结果回填到 `Config.KEY_LAYER_MAP`。

## 数据流

核心流程如下：

`build_federated_datasets -> build_model -> train_local_model -> attack adapter -> delta -> layer scoring -> selection`

含义是：

1. 用当前 `MODEL / DATASET / PARTITION` 构造数据与模型元信息。
2. 服务器取纯净代理数据集，模拟 benign / malicious 两条本地训练分支。
3. 若攻击包含数据投毒，则先复用现有 attack adapter 改写恶意分支数据。
4. 若攻击包含 update poisoning，则训练完成后再复用现有 adapter 构造恶意上传模型。
5. 客户端上传对象始终视为**训练后的完整模型参数**，再通过 `local_state_dict - global_state_dict` 得到更新量。
6. 逐层计算扰动得分、做时序聚合与跨攻击共识，最终输出关键层。

## 层粒度定义

候选层默认只包含**带参数的叶子 `Conv2d / Linear` 模块**。

这样做有两个目的：

- 与现有 `KEY_LAYER_MAP` / `selected_layers` 的前缀用法保持一致。
- 避开 BatchNorm 统计量与非线性层，减少“状态量变化大但不一定有防御价值”的噪声。

## 算法实现

### Step 1: 本地模拟

对每个攻击 `j` 和每一轮 `t`：

1. 从同一个全局模型 `w^(t-1)` 出发。
2. 训练 1 个良性控制分支，得到 benign 上传模型。
3. 训练 1 个恶意实验分支，必要时叠加数据投毒或 update 投毒，得到 malicious 上传模型。
4. 下一轮全局模型默认沿**良性轨迹**推进，即 `w^(t) <- w_B^(t)`。

对于 `ALIE / FedImp` 这类依赖良性群体统计的攻击，工具会额外构造 benign reference pool，并继续复用现有 adapter 的 `benign_states` 语义，而不是把它们错误降格成普通 1vs1 攻击。

### Step 2: 逐层特征提取

对每个候选层 `l`：

- `Δw_B,l^(t)`：良性上传模型相对当前全局模型的该层更新
- `Δw_M,l^(t)`：恶意上传模型相对当前全局模型的该层更新

计算：

`m_l^(t) = ||Δw_M,l^(t) - Δw_B,l^(t)||_2 / sqrt(d_l)`

`c_l^(t) = 1 - cos(Δw_M,l^(t), Δw_B,l^(t))`

其中 `d_l` 是该层参数维度。若 benign 和 malicious 在该层都是近零更新，则直接令 `c_l^(t) = 0`，避免零向量余弦距离带来伪异常。

### Step 3: 跨层 Z-score

在当前轮全部候选层上分别做标准化：

`hat(m)_l^(t) = (m_l^(t) - mu_m^(t)) / (sigma_m^(t) + eps)`

`hat(c)_l^(t) = (c_l^(t) - mu_c^(t)) / (sigma_c^(t) + eps)`

这一步的作用是做**无量纲对齐**，让幅度特征和方向特征可直接组合。

### Step 4: 自适应赋权

当前实现**不会**按 `Var(hat(m)) / Var(hat(c))` 计算权重，而是采用下面这组公式：

`sigma^2_(m,t) = Var({m_l^(t)})`

`sigma^2_(c,t) = Var({c_l^(t)})`

`alpha_t = sigma^2_(m,t) / (sigma^2_(m,t) + sigma^2_(c,t) + eps)`

`beta_t = 1 - alpha_t`

`s_l^(t) = alpha_t * hat(m)_l^(t) + beta_t * hat(c)_l^(t)`

#### 为什么要这样改

如果先做 Z-score，再对 `hat(m)` 和 `hat(c)` 取跨层方差，那么这两个方差在大多数轮次都会非常接近 1。

结果就是：

- `alpha_t` 与 `beta_t` 会退化到接近 `0.5 / 0.5`
- “自适应赋权”在数值上几乎失效

所以当前实现采用：

- **Z-score 负责无量纲对齐**
- **原始跨层方差负责评估哪一类特征在本轮更有区分度**

这是一个**有意修正**，不是实现偏差。`selection.json` 中会固定记录：

- `weighting_mode = "raw_variance_on_zscored_scores"`

## Step 5: 时序聚合与跨攻击共识

对每个攻击 `j`：

`S_l^(j) = (1 / T) * sum_t s_l^(t)`

再做跨攻击共识：

`V_l = sum_j max(0, S_l^(j))`

最终选层规则：

1. 直接按 `V_l` 从高到低排序。
2. 取前 `K` 个层作为最终关键层。
3. 若分数并列，则按候选层顺序稳定打破平局。

## 模块说明

- `settings.py`
  提取专用配置
- `candidates.py`
  候选层生成
- `simulator.py`
  benign / malicious 模拟
- `scoring.py`
  层级得分与时序/共识聚合
- `selection.py`
  最终关键层选择
- `export.py`
  结果导出
- `pipeline.py`
  总调度入口

## 输出文件

### `selection.json`

主要字段：

- `selected_layers`
- `candidate_layers`
- `top1_by_attack`
- `dropped_top1_by_attack`
- `per_attack_scores`
- `consensus_scores`
- `weighting_mode`
- `config_key_layer_map_entry`

其中 `config_key_layer_map_entry` 可以直接拷到 `Config.KEY_LAYER_MAP` 中对应模型的条目里。

### `layer_scores.csv`

每行一个候选层，包含：

- 是否被选中
- 选中顺序
- 综合共识分数
- 各个攻击下的 `S_l^(j)`
- 哪些攻击把该层当作 Top-1

## 用法

默认运行：

```bash
python -m layer_extraction
```

如果只想做快速 smoke test，可以在代码里构造 `LayerExtractionSettings`，例如缩短轮数或限制代理数据样本数，但默认语义仍然是使用**全量干净训练集**作为代理数据。

## 模型选择配置

现在可以通过 `LayerExtractionSettings.selected_models` 一次指定要跑哪些模型。

批量模型模式下，数据集映射固定为：

- `lenet5 -> mnist`
- 其余模型 -> `cifar10`

也就是说，只需要选模型，不需要再给每个模型单独配数据集。

示例：

```python
from config import Config
from layer_extraction import LayerExtractionSettings, run_layer_extraction

settings = LayerExtractionSettings.from_config(
    base_config=Config,
    selected_models=["lenet5", "resnet20", "resnet18"],
)

run_layer_extraction(settings)
```

如果不传 `selected_models`，工具仍然沿用单模型模式，默认读取当前 `config.py` 里的 `MODEL / DATASET / PARTITION`。

## 已有结果上自动选 K

如果已经跑过离线层提取，不需要重新训练即可用肘部法则估计 `k`：

```bash
python -m layer_extraction.select_k
```

这个命令默认读取：

```text
results/layer_extraction/*/selection.json
results/layer_extraction/*_plot_data.csv
```

也可以直接读取已经整理好的绘图数据：

```bash
python -m layer_extraction.select_k layer_extraction/select_k/layer_extraction_data
```

默认 `chord` 方法会在降序 `consensus_scores` 曲线上找离首尾连线最远的 rank，并把该 rank 作为自动 `k`。如果想使用相邻分数最大下降，也可以运行：

```bash
python -m layer_extraction.select_k --method max_gap
```

为了避免第一层特别强时退化成只选 1 层，可以限制搜索范围：

```bash
python -m layer_extraction.select_k --min-k 2 --max-k 5
```

如果需要同步画出每个模型的肘部曲线：

```bash
python -m layer_extraction.select_k layer_extraction/select_k/layer_extraction_data \
  --plot-dir layer_extraction/select_k/layer_extraction_data/select_k_plots
```

如果需要五个模型合在一起的论文版 PDF：

```bash
python -m layer_extraction.select_k layer_extraction/select_k/layer_extraction_data \
  --combined-pdf layer_extraction/select_k/layer_extraction_data/select_k_elbow_combined.pdf
```

可以用 `--force-k lenet5=1` 对小模型施加论文图中的人工预算约束。
