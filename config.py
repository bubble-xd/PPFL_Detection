from __future__ import annotations

import torch


class Config:
    # ----------------------------
    # Reproducibility / runtime
    # ----------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_ROOT = "data"
    RESULTS_ROOT = "results"
    # 运行时性能开关：CUDA 可用时启用 TF32/AMP/channels_last，CPU 环境会自动退化。
    ENABLE_TF32 = True
    CUDNN_BENCHMARK = True
    USE_AMP = True
    USE_CHANNELS_LAST = True
    PIN_MEMORY = None
    DATA_LOADER_NUM_WORKERS = 0
    DATA_LOADER_PERSISTENT_WORKERS = False
    DATA_LOADER_PREFETCH_FACTOR = None
    # 服务端特征矩阵、距离矩阵和 FedImp dense 统计优先放到 GPU；超过预算自动回退 CPU。
    SERVER_COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SERVER_FEATURE_GPU_MAX_MB = 512
    TORCH_NUM_THREADS = None
    TORCH_NUM_INTEROP_THREADS = None
    SAVE_ROUND_LOGS = False
    SAVE_CSV_LOGS = False
    EXPORT_EXCEL = True
    # 控制 Excel 导出的指标页："all" 全部保存，"bm_gap" 只保存 BMGap，"acc_asr" 只保存 ACC 和 ASR。
    EXCEL_METRIC_SHEETS = "bm_gap"
    PRINT_PROGRESS = True
    SAVE_TEXT_LOG = True
    TEXT_LOG_FILENAME = "run.log"
    LOG_CLIENT_UPDATES = True
    # ASR 容易受最后一轮偶然波动影响
    ASR_SUMMARY_TAIL_ROUNDS = 10
    CLIENT_UPDATE_PREVIEW_VALUES = 6
    SAVE_COSINE_HEATMAPS = True
    COSINE_HEATMAP_ROUNDS = None  # None -> [1, mid, last]

    # ----------------------------
    # FL training
    # ----------------------------
    NUM_CLIENTS = 20
    NUM_ROUNDS = 30
    LOCAL_EPOCHS = 1

    MODEL = "resnet34"
    DATASET = "cifar100"
    # MODEL = "lenet5"
    # DATASET = "mnist"

    BATCH_SIZE = 64
    LR = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    OPTIMIZER = "sgd"
    MAX_BATCHES = 300

    ROBUST_METHODS = ["krum","multi_krum","median","clustering"] #"krum","multi_krum","median","clustering","fedavg"
    # 支持单个投毒比例，也支持批量实验列表，例如 `[0.1, 0.2, 0.3, 0.4]`。
    POISON_RATE =[0.2]
    MALICIOUS_CLIENT_SELECTION_SEED = 123
    # ----------------------------
    # Experiment selection
    # 一次配置一个模型、一个数据集、一种划分
    # `resnet18` / `resnet34` / `vgg11` 当前主要按 CIFAR10 场景接入
    # ----------------------------
    PARTITION = "iid"  # "iid" or "noniid"

    # ----------------------------
    # Attack selection
    # ----------------------------
    SELECTED_ATTACKS = [
        "badnets",
        "label_flipping_targeted",
        "edge_case",
        "dba",
        "semantic_backdoor",
        "label_flipping_untargeted",
        "additive_noise",
        "scaling_attack",
        "sign_flipping",
        "random_gradient",
        "a_lie",
        "fedimp",
    ]

    # ----------------------------
    # Attack strength
    # 每种攻击的可调强度统一放这里
    # ----------------------------
    ATTACK_STRENGTHS = {
        "badnets": {"poisoning_ratio": 1.0, "target_label": 1, "trigger_size": 3},
        "dba": {"poisoning_ratio": 1.0, "target_label": 1, "trigger_size": 3},
        "edge_case": {
            "poisoning_ratio":1.0,
            "epsilon": 0.5,
            "projection_type": "l_2",
            "scaling_factor": 25.0,
        },
        "semantic_backdoor": {
            "poisoning_ratio": 0.3,
            "epsilon": 0.5,
            "projection_type": "l_2",
            "scaling_factor": 25.0,
        },
        #cifar100:
        "label_flipping_targeted": {
        "source_class": 42,  # leopard
        "target_class": 88,  # tiger
        "poison_ratio": 1.0,
        },
        #cifar-10:
        #"label_flipping_targeted": {
        #     "source_class": 3,
        #     "target_class": 5,
        #     "poison_ratio": 1.0,
        # },
        "label_flipping_untargeted": {"poison_ratio": 0.5},
        "sign_flipping": {"alpha": 0.25},
        "scaling_attack": {"gamma": 2},
        "additive_noise": {"mean": 0.0, "std": 0.05},
        "random_gradient": {"mean": 0.0, "std": 0.05},
        "a_lie": {"z_max": 0.5, "client_jitter_std": 0.05},   # 0.0 disables per-client jitter
        "fedimp": {
            "fedimp_factor": 0.8,
            "top_k_ratio": 0.1,
            "compute_device": "cuda" if torch.cuda.is_available() else "cpu",
            "dense_stats_max_mb": 512,
        },
    }

    # 数据集相关的攻击参数覆盖
    ATTACK_STRENGTHS_BY_DATASET = {
        "edge_case": {
            "mnist": {"target_label": 1},
            "cifar10": {"target_label": 9},
            "cifar100": {"target_label": 99},
        },
        "semantic_backdoor": {
            "mnist": {"target_label": 1, "semantic_source": "ardis"},
            "cifar10": {"target_label": 9, "semantic_source": "southwest"},
            "cifar100": {"target_label": 99, "semantic_source": "southwest"},
        },
    }
    # VGG11 是 1.29e8 量级的大模型，大特征模式会额外触发内存安全路径；
    # 下面几个预算参数用于限制中间特征矩阵 / 投影矩阵的显式物化大小。
    FEATURE_MODES = [
        "raw_full",
       # "selected_layers",
       "selected_layers_balanced",
       # "selected_layers_projected",
        "selected_layers_balanced_projected",
    ]  # , "control_layer"
    PROJECTION_SEED = 2026
    PROJECTION_DIM = 2048
    FEATURE_MATRIX_MAX_MB = 2048
    PROJECTION_MATRIX_MAX_MB = 256
    FEATURE_STREAM_CHUNK_SIZE = 65536

    # 只有超大模型默认走“客户端 state 落盘 / 顺序聚合”；
    DISK_STATE_CACHE_MODELS = ["vgg11"]

    # ----------------------------
    # Data partition
    # iid: standard IID split
    # noniid: Dirichlet non-IID split
    # ----------------------------
    DIRICHLET_ALPHA = 1.0
    PARTITION_PRESETS = {
        "iid": {"type": "iid"},
        "noniid": {"type": "dirichlet", "alpha": DIRICHLET_ALPHA},
    }

    # ----------------------------
    # Robust aggregation hyper-parameters
    # ----------------------------
    KRUM_F = None
    MULTI_KRUM_F = None
    MULTI_KRUM_M = None
    GEOM_MEDIAN_MAX_ITERS = 100
    GEOM_MEDIAN_TOL = 1e-5
    KMEANS_RANDOM_STATE = 42
    KMEANS_N_INIT = 10

    METHOD_DISPLAY_NAMES = {
        "fedavg": "FedAvg",
        "krum": "Krum",
        "multi_krum": "Multi-Krum",
        "median": "Median",
        "clustering": "Clustering",
    }
    FEATURE_DISPLAY_NAMES = {
        "raw_full": "原始",
        "selected_layers": "提取",
        "selected_layers_balanced": "提取+平衡",
        "selected_layers_projected": "提取+投影",
        "selected_layers_balanced_projected": "提取+平衡+投影",
        "control_layer": "对照层",
    }
# ['fc', 'layer3.2.conv2', 'conv1', 'layer3.2.conv1', 'layer3.0.conv2']
    KEY_LAYER_MAP = {
        "lenet5": ["fc3"],
        "resnet20": ['fc', 'layer3.2.conv2', 'conv1', 'layer3.2.conv1', 'layer3.0.conv2'],
         "resnet18": ["fc", "layer4.1.conv2"],#, "layer4.1.conv2", "conv1", "layer4.1.conv1", "layer3.1.conv1"
        "resnet34": ['fc', 'layer4.2.conv2', 'conv1', 'layer4.2.conv1'],
        "vgg11":['features.0', 'classifier.3', 'classifier.0'],
    }
    # 对照层刻意选在“尽量靠前、但不与关键层重叠”的卷积层：
    # 这类低层特征更偏向通用边缘/纹理，通常比后段语义层更不容易被投毒攻击直接牵引。
    CONTROL_LAYER_MAP = {
        "lenet5": ["conv1"],
        "resnet20": ["layer1.0.conv1"],
        "resnet18": ["layer1.0.conv1"],
        "resnet34": ["layer1.0.conv1"],
        "vgg11": ["features.0"],
    }
    MODEL_DISPLAY_NAMES = {
        "lenet5": "LeNet5",
        "resnet20": "ResNet20",
        "resnet18": "ResNet18",
        "resnet34": "ResNet34",
        "vgg11": "VGG11",
    }
    DATASET_DISPLAY_NAMES = {
        "mnist": "MNIST",
        "cifar10": "CIFAR10",
        "cifar100": "CIFAR100",
    }

    @classmethod
    def get_device(cls) -> str:
        if cls.DEVICE == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return cls.DEVICE

    @classmethod
    def get_server_compute_device(cls) -> str:
        server_device = str(getattr(cls, "SERVER_COMPUTE_DEVICE", cls.get_device())).strip().lower()
        if server_device in {"auto", "cuda"} and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @classmethod
    def _normalize_name(cls, value: str) -> str:
        return str(value).strip()

    @classmethod
    def _resolve_partition(cls, partition_name: str | dict) -> dict:
        if isinstance(partition_name, dict):
            return dict(partition_name)

        key = cls._normalize_name(partition_name).lower()
        if key not in cls.PARTITION_PRESETS:
            raise ValueError(f"Unsupported partition preset: {partition_name}")
        return dict(cls.PARTITION_PRESETS[key])

    @classmethod
    def _format_experiment_name(cls, model: str, dataset: str, partition_name: str) -> str:
        partition_key = cls._normalize_name(partition_name).lower()
        partition_label = "IID" if partition_key == "iid" else "NonIID"
        model_label = cls.MODEL_DISPLAY_NAMES.get(model, model)
        dataset_label = cls.DATASET_DISPLAY_NAMES.get(dataset, dataset.upper())
        return f"{model_label}-{dataset_label}-{partition_label}"

    @classmethod
    def get_enabled_experiments(cls):
        model_name = cls._normalize_name(cls.MODEL).lower()
        dataset_name = cls._normalize_name(cls.DATASET).lower()
        partition_name = cls.PARTITION
        partition_key = (
            cls._normalize_name(partition_name).lower()
            if isinstance(partition_name, str)
            else cls._normalize_name(partition_name.get("type", "iid")).lower()
        )
        return [
            {
                "name": cls._format_experiment_name(
                    model=model_name,
                    dataset=dataset_name,
                    partition_name=partition_key,
                ),
                "model": model_name,
                "dataset": dataset_name,
                "partition": cls._resolve_partition(partition_name),
            }
        ]

    @classmethod
    def get_enabled_attack_configs(cls):
        attack_configs = []
        for attack_name in cls.SELECTED_ATTACKS:
            name = cls._normalize_name(attack_name).lower()
            attack_configs.append(
                {
                    "name": name,
                    "params": dict(cls.ATTACK_STRENGTHS.get(name, {})),
                    "params_by_dataset": dict(cls.ATTACK_STRENGTHS_BY_DATASET.get(name, {})),
                }
            )
        return attack_configs

    @classmethod
    def get_poison_rates(cls) -> list[float]:
        raw_poison_rate = getattr(cls, "POISON_RATE", 0.0)
        raw_values = (
            list(raw_poison_rate)
            if isinstance(raw_poison_rate, (list, tuple))
            else [raw_poison_rate]
        )
        if not raw_values:
            raise ValueError("POISON_RATE 列表不能为空。")

        normalized_rates: list[float] = []
        for raw_value in raw_values:
            poison_rate = float(raw_value)
            if not 0.0 <= poison_rate <= 1.0:
                raise ValueError(f"POISON_RATE 必须落在 [0, 1] 区间内，收到 {raw_value!r}")
            normalized_rates.append(poison_rate)
        return normalized_rates

    @classmethod
    def get_num_malicious(cls, poison_rate: float | None = None) -> int:
        # 当 `POISON_RATE` 配成列表时，这里默认取第一项，
        # 方便依赖旧接口的代码在单轮上下文里继续工作。
        resolved_poison_rate = (
            float(poison_rate)
            if poison_rate is not None
            else float(cls.get_poison_rates()[0])
        )
        num_malicious = int(cls.NUM_CLIENTS * resolved_poison_rate)
        if resolved_poison_rate > 0 and num_malicious == 0:
            num_malicious = 1
        return min(cls.NUM_CLIENTS, max(0, num_malicious))

    @classmethod
    def get_multi_krum_m(
        cls,
        default_num_malicious: int | None = None,
        poison_rate: float | None = None,
    ) -> int:
        if cls.MULTI_KRUM_M is not None:
            return int(cls.MULTI_KRUM_M)
        resolved_num_malicious = (
            int(default_num_malicious)
            if default_num_malicious is not None
            else cls.get_num_malicious(poison_rate=poison_rate)
        )
        return max(1, cls.NUM_CLIENTS - resolved_num_malicious - 2)

    @classmethod
    def get_krum_f(cls, default_num_malicious: int) -> int:
        if cls.KRUM_F is not None:
            return int(cls.KRUM_F)
        return int(default_num_malicious)

    @classmethod
    def get_multi_krum_f(cls, default_num_malicious: int) -> int:
        if cls.MULTI_KRUM_F is not None:
            return int(cls.MULTI_KRUM_F)
        return int(default_num_malicious)

    @classmethod
    def should_stream_client_states(cls, model_name: str) -> bool:
        normalized_model = cls._normalize_name(model_name).lower()
        enabled_models = {
            cls._normalize_name(enabled_model).lower()
            for enabled_model in getattr(cls, "DISK_STATE_CACHE_MODELS", [])
        }
        return normalized_model in enabled_models
