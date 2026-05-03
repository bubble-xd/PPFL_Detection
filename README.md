# PPFL Detection

这份 README 只解释代码结构、模块接口和整体流程，不包含环境安装或启动说明。

## 整体流程

项目的主线可以概括成一条固定的数据流：

```text
Config
  -> run_all_experiments
  -> build_federated_datasets
  -> build_attack_adapter
  -> 每轮为每个客户端训练本地模型
  -> 收集 local_state_dict
  -> FeatureBuilder.build_feature_matrix
  -> run_aggregator
  -> 更新 global_state
  -> evaluate_accuracy / evaluate_asr / compute_detection_f1
  -> export_experiment_results
```

如果按“读代码”的顺序看，推荐这样理解：

1. `config.py` 定义实验参数和派生配置。
2. `main.py` 把 `Config` 交给 `utils/benchmark.py`。
3. `utils/benchmark.py` 负责组织整套实验流程。
4. `data/` 生成联邦数据。
5. `attacks/` 决定如何投毒数据或篡改客户端本地模型。
6. `utils/training.py` 完成本地训练。
7. `utils/state_dict.py` 提供本地模型与差分向量之间的内部转换工具。
8. `features/` 从本地模型提取聚合器需要的特征。
9. `aggregators/` 做鲁棒聚合并输出检测结果。
10. `metrics/` 计算 `ACC / ASR / F1`。
11. `utils/io.py` 把结果导出到 `results/`。

## 顶层模块

| 模块 | 作用 | 关键接口 |
| --- | --- | --- |
| `main.py` | 项目总入口 | `main()` |
| `config.py` | 所有实验配置与配置派生逻辑 | `Config` |
| `data/` | 加载数据、切分客户端数据 | `build_federated_datasets()`、`partition_dataset()` |
| `models/` | 构建模型 | `build_model()` |
| `attacks/` | 构建攻击适配器，统一数据投毒和模型投毒接口 | `build_attack_adapter()` |
| `features/` | 从客户端本地模型中构造特征矩阵 | `FeatureBuilder.build_feature_matrix()` |
| `aggregators/` | 鲁棒聚合与恶意客户端识别 | `run_aggregator()` |
| `metrics/` | 评估分类精度、ASR、检测 F1 | `evaluate_accuracy()`、`evaluate_asr()`、`compute_detection_f1()` |
| `utils/benchmark.py` | 调度整个实验 | `run_all_experiments()` |
| `utils/state_dict.py` | 本地模型聚合与重建工具 | `average_tensor_dicts()`、`reconstruct_state_dict_like()` |
| `utils/training.py` | 本地训练 | `train_local_model()` |
| `utils/io.py` | 结果目录创建与表格导出 | `create_run_directory()`、`export_experiment_results()` |

## 核心对象与统一契约

理解这个项目，先抓住 5 个核心对象。

### 1. `Config`

位置：`config.py`

这是全局配置类，既保存静态参数，也负责生成实验描述。

核心方法：

- `Config.get_enabled_experiments() -> List[dict]`
  返回当前启用的实验列表。每个实验字典至少包含：
  - `name`
  - `model`
  - `dataset`
  - `partition`
- `Config.get_enabled_attack_configs() -> List[dict]`
  返回当前启用的攻击配置。每个攻击字典至少包含：
  - `name`
  - `params`
  - `params_by_dataset`
- `Config.get_num_malicious() -> int`
  根据 `POISON_RATE` 计算恶意客户端数量。
- `Config.get_krum_f(default_num_malicious) -> int`
- `Config.get_multi_krum_f(default_num_malicious) -> int`
- `Config.get_multi_krum_m() -> int`

`Config` 的角色不是“保存常量”这么简单，而是把“可读配置”转成“实验调度器可直接消费的结构”。

### 2. `FederatedDataBundle`

位置：`data/data_loader.py`

这是联邦数据构建后的统一封装对象，字段包括：

- `dataset_name`
- `train_dataset`
- `test_dataset`
- `client_datasets`
- `test_loader`
- `dataset_info`
- `partition_info`

其中最关键的是：

- `client_datasets`：`Dict[int, TensorDataset]`，每个客户端一份本地数据
- `dataset_info`：模型构建和攻击初始化需要的元信息
- `partition_info`：记录当前是 `iid` 还是 `dirichlet`

### 3. `local_state_dict`

位置：客户端本地训练完成后产生

这是客户端真正提交给服务端的对象。它表示“本地训练后的完整模型参数”，本质上是：

- `Dict[str, Tensor]`

键是参数名，值是本地模型对应参数。FedAvg、Krum、几何中位数、聚类筛选等聚合器，都是直接围绕 `local_state_dict` 工作。

### 4. `update_dict`

位置：主要由 `utils/state_dict.py` 在服务端内部按需产生

这是一个内部中间对象。它表示“本地模型相对全局模型的参数更新”，本质上是：

- `Dict[str, Tensor]`

键是参数名，值是浮点张量增量。它主要用于更新投毒攻击、诊断日志，以及少量分析逻辑。

### 5. `feature_matrix`

位置：由 `features/FeatureBuilder` 产生

这是聚合器用来比较客户端相似性的二维矩阵：

- 形状通常是 `[num_clients, feature_dim]`

不同 `feature_mode` 会决定从哪些层提取本地模型参数，以及是否做随机投影。

### 6. `AggregationResult`

位置：`aggregators/common.py`

所有聚合器的统一返回类型：

- `aggregated_state`
- `predicted_malicious_ids`
- `selected_client_ids`
- `aux_scores`

含义分别是：

- `aggregated_state`：最终聚合得到的新全局模型参数
- `predicted_malicious_ids`：该聚合器判定的恶意客户端 id
- `selected_client_ids`：真正参与聚合的客户端 id
- `aux_scores`：附加分数，如 Krum 分数或聚类标签

## 各模块与接口

### `main.py`

这是最薄的一层封装。

核心接口：

- `main() -> None`

职责只有一个：把 `Config` 交给 `run_all_experiments(Config)`。

### `config.py`

这是实验的配置中心。除了常见超参数，还定义了：

- 模型选择：`MODEL`
- 数据集选择：`DATASET`
- 划分方式：`PARTITION`
- 攻击列表：`SELECTED_ATTACKS`
- 攻击强度：`ATTACK_STRENGTHS`
- 特征模式：`FEATURE_MODES`
- 鲁棒聚合器：`ROBUST_METHODS`
- 层选择映射：`KEY_LAYER_MAP`、`CONTROL_LAYER_MAP`

这里还有几个很重要的语义：

- `POISON_RATE`：恶意客户端占比
- `poisoning_ratio` / `poison_ratio`：恶意客户端本地数据或标签被修改的比例

这两个量控制的是不同层面。

### `data/`

#### `data/data_loader.py`

负责把原始数据集转换成联邦学习可直接使用的对象。

核心接口：

- `build_federated_datasets(dataset_name, num_clients, partition, batch_size, root) -> FederatedDataBundle`
  加载原始数据集、切分客户端数据、构造测试集与测试加载器。
- `materialize_dataset(dataset) -> TensorDataset`
  把任意 `Dataset` 实体化成内存中的 `TensorDataset`。

#### `data/fl_partition.py`

负责切分策略。

核心接口：

- `partition_dataset(labels, num_clients, partition) -> Dict[str, object]`
  根据配置选择具体切分方法。
- `partition_iid(...) -> Dict[int, List[int]]`
- `partition_dirichlet(...) -> Dict[int, List[int]]`

输出里最重要的是 `client_indices`，它定义了每个客户端拿到哪些样本。

### `models/`

模型模块分成“具体模型定义”和“统一模型工厂”两层。

核心接口：

- `build_model(model_name, input_channels, num_classes, image_size)`
- `get_model_names() -> List[str]`

具体模型：

- `models/lenet5.py` 中的 `LeNet5`
- `models/resnet20.py` 中的 `resnet20()` 与 `ResNet`
- `models/resnet.py` 中的 `resnet18()`、`resnet34()`
- `models/vgg11.py` 中的 `vgg11()`

外部代码不直接关心具体类名，统一通过 `build_model()` 获取模型实例。

### `attacks/`

这个模块的关键不是具体攻击本身，而是“统一适配接口”。

#### `attacks/registry.py`

核心接口：

- `build_attack_adapter(attack_config, dataset_name, dataset_info)`
  根据攻击名字和数据集信息构建攻击适配器。
- `get_enabled_attack_configs(attack_configs)`
  过滤启用的攻击配置。

#### `attacks/adapters.py`

定义了统一的攻击适配器接口：`BaseAttackAdapter`。

核心字段：

- `attack_name`
- `attack_mode`
- `is_data_poisoning`
- `is_update_poisoning`

核心方法：

- `poison_local_dataset(dataset, client_id, malicious_ids)`
  数据投毒攻击使用，返回投毒后的本地数据集。
- `poison_client_state(current_state, client_id, malicious_ids, benign_states, global_state_dict, num_clients)`
  模型投毒攻击使用，返回投毒后的本地模型参数。
- `build_asr_eval_loader(clean_test_dataset, batch_size)`
  目标攻击使用，为 ASR 评估构造测试数据。

也就是说，`utils/benchmark.py` 并不需要知道某个攻击的内部细节，只需要调用统一接口。

#### `attacks/targeted/`

这里存放目标攻击实现。代表性接口包括：

- `BadNetsAttack`
- `DBAAttack`
- `EdgeCaseAttack`
- `SemanticBackdoorAttack`
- `label_flipping_targeted(...)`

这些实现最终都会被适配到 `BaseAttackAdapter` 接口上。

#### `attacks/untargeted/`

这里存放无目标攻击实现。代表性接口包括：

- `label_flipping_untargeted(...)`
- `sign_flipping(...)`
- `scaling_attack(...)`
- `additive_noise_attack(...)`
- `random_gradient_attack(...)`
- `alie_attack_update(...)`
- `fedimp_attack_update(...)`

这些函数通常在内部操作 `update_dict` 或标签张量，但不会改变“客户端对外提交的是本地模型”这一主语义。

### `utils/training.py`

负责单个客户端的本地训练。

核心接口：

- `build_loader(dataset, batch_size, shuffle, seed) -> DataLoader`
- `train_local_model(model, dataset, device, local_epochs, batch_size, learning_rate, momentum, weight_decay, seed, optimizer_name, max_batches)`

`train_local_model()` 的输入是“当前轮的全局模型副本 + 某个客户端的本地数据”，输出是训练后的本地模型。

### `utils/state_dict.py`

这是训练、特征提取、聚合器之间的桥梁模块。

核心接口：

- `clone_state_dict(state_dict, device=None) -> Dict[str, Tensor]`
- `average_tensor_dicts(tensor_dicts, selected_ids=None, reference_state_dict=None) -> Dict[str, Tensor]`
- `flatten_tensor_dict(tensor_dict, keys=None) -> Tensor`
- `reconstruct_state_dict_like(flat_tensor, reference_state_dict, keys=None) -> Dict[str, Tensor]`
- `select_tensor_dict_by_prefixes(tensor_dict, prefixes) -> Dict[str, Tensor]`

一句话理解：

- 客户端训练结束后，先收集完整 `local_state_dict`
- 只有在更新投毒阶段，才在攻击适配层内部临时计算模型差值

### `features/`

这个模块把 `local_state_dict` 变成聚合器可比较的二维特征矩阵。

#### `features/extractors.py`

核心类：

- `FeatureBuilder`

核心方法：

- `get_selected_prefixes() -> List[str]`
- `get_control_prefixes() -> List[str]`
- `build_feature_matrix(local_state_dicts, feature_mode) -> Tensor`

支持的特征模式：

- `raw_full`
- `selected_layers`
- `selected_layers_projected`
- `control_layer`

#### `features/projection.py`

核心接口：

- `build_orthogonal_projection(input_dim, output_dim, seed) -> Tensor`

用于 `selected_layers_projected` 模式中的随机正交投影。

### `aggregators/`

聚合器模块有统一入口，也有具体实现。

#### `aggregators/__init__.py`

统一入口：

- `run_aggregator(method_name, local_state_dicts, feature_matrix, num_malicious, global_state_dict, config) -> AggregationResult`

外部调度器只需要知道方法名，不需要直接调用具体聚合器函数。

#### `aggregators/common.py`

提供公共类型和数学工具：

- `AggregationResult`
- `pairwise_squared_distances(feature_matrix)`
- `geometric_median(points, max_iters, tol)`
- `aggregate_geometric_median(local_state_dicts, reference_state_dict, max_iters, tol)`
- `aggregate_mean(local_state_dicts, reference_state_dict, selected_client_ids=None)`

#### 具体聚合器

- `fedavg_aggregate(local_state_dicts, global_state_dict) -> AggregationResult`
  不做检测，直接平均所有客户端本地模型。
- `krum_aggregate(local_state_dicts, feature_matrix, num_malicious, global_state_dict) -> AggregationResult`
  用 Krum 分数选出一个客户端本地模型作为聚合结果。
- `multi_krum_aggregate(local_state_dicts, feature_matrix, num_malicious, m, global_state_dict) -> AggregationResult`
  选出多个客户端后再平均。
- `median_aggregate(local_state_dicts, feature_matrix, num_malicious, global_state_dict, max_iters, tol) -> AggregationResult`
  通过几何中位数完成聚合，并用到中心距离识别异常点。
- `clustering_aggregate(local_state_dicts, feature_matrix, global_state_dict, random_state, n_init) -> AggregationResult`
  先聚类，再把可疑簇排除后做平均。

### `metrics/`

指标模块很直观，负责把模型表现和检测表现量化。

核心接口：

- `evaluate_accuracy(model, data_loader, device) -> float`
- `evaluate_asr(model, data_loader, device) -> float`
- `compute_detection_f1(predicted_ids, malicious_ids, num_clients) -> float`

这三个指标分别对应：

- `ACC`：干净测试集精度
- `ASR`：目标攻击成功率
- `F1`：恶意客户端检测效果

### `utils/benchmark.py`

这是全项目最核心的调度模块。

核心接口：

- `run_all_experiments(config) -> str`

它负责：

1. 创建结果目录和日志器
2. 从 `Config` 取出实验与攻击配置
3. 构建联邦数据集
4. 为每种攻击构建攻击适配器
5. 为每种聚合器和每种特征模式执行完整训练流程
6. 汇总每轮结果
7. 导出 `csv / xlsx / run.log`

内部最重要的私有函数是：

- `_run_single_setting(...)`

它表示“固定一个实验、一个攻击、一个聚合器、一个特征模式”时的完整单次流程。

### `utils/io.py`

负责把内存中的实验结果整理成文件。

核心接口：

- `create_run_directory(results_root) -> str`
- `export_experiment_results(output_dir, experiment_name, summary_records, round_logs, attacks, methods, feature_modes, method_display_names, feature_display_names, save_csv_logs, save_round_logs, export_excel) -> None`

可以把它理解成“结果展示层”。

### 其他辅助模块

#### `utils/random.py`

- `set_global_seed(seed, deterministic=True)`
- `derive_seed(base_seed, *parts)`

负责整个实验中的可复现性。

#### `utils/logger.py`

- `RunLogger`

同时负责控制台输出和文本日志落盘。

## 一轮实验内部到底发生了什么

如果只关心“每轮怎么跑”，可以直接看这条链：

1. 取出当前轮的全局模型参数。
2. 遍历每个客户端。
3. 如果该客户端是恶意客户端，且攻击属于数据投毒，则先调用 `poison_local_dataset(...)`。
4. 用 `train_local_model(...)` 在本地数据上训练客户端模型。
5. 收集所有客户端的 `local_state_dict`。
6. 如果攻击属于更新投毒，则在服务端内部按需转成 update，生成投毒后的本地模型。
7. 用 `FeatureBuilder.build_feature_matrix(...)` 构造特征矩阵。
8. 用 `run_aggregator(...)` 得到 `AggregationResult`。
9. 直接使用 `aggregated_state` 作为新的全局模型参数。
10. 只在攻击内部需要时，才临时计算 `update_dict`。
11. 用 `evaluate_accuracy()`、`evaluate_asr()`、`compute_detection_f1()` 记录指标。

项目每一轮的逻辑基本都围绕这 11 步展开。

## 读代码时最值得注意的三个接口边界

### 1. 攻击层和调度层的边界

`utils/benchmark.py` 不直接关心具体攻击是 BadNets 还是 sign flipping，它只认 `BaseAttackAdapter` 暴露的统一方法。

### 2. 训练层和聚合层的边界

训练层输出的是模型，聚合层消费的也是本地模型。只有攻击层才按需使用 `update_dict`。

### 3. 特征层和聚合层的边界

聚合器并不关心“关键层是哪些层”，这些都由 `FeatureBuilder` 先处理成统一的 `feature_matrix`。

## 推荐阅读顺序

如果你要快速看懂全仓库，建议按下面的顺序打开文件：

1. `config.py`
2. `main.py`
3. `utils/benchmark.py`
4. `data/data_loader.py`
5. `attacks/adapters.py`
6. `utils/training.py`
7. `utils/state_dict.py`
8. `features/extractors.py`
9. `aggregators/common.py`
10. `aggregators/krum.py`、`aggregators/median.py`、`aggregators/clustering.py`
11. `metrics/classification.py`、`metrics/detection.py`

按这个顺序读，最容易把“配置 -> 数据 -> 攻击 -> 训练 -> 特征 -> 聚合 -> 指标 -> 导出”这条主线串起来。
