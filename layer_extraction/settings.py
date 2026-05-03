from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import torch

from config import Config


def _default_k_by_model(base_config) -> Dict[str, int]:
    # 默认预算直接沿用当前 KEY_LAYER_MAP 的层数，避免离线工具和主流程脱节。
    return {
        str(model_name).strip().lower(): len(list(layer_prefixes))
        for model_name, layer_prefixes in dict(getattr(base_config, "KEY_LAYER_MAP", {})).items()
    }


@dataclass
class LayerExtractionSettings:
    base_config: object = Config
    model: Optional[str] = None
    dataset: Optional[str] = None
    selected_models: Optional[List[str]] = None
    partition: Optional[str | dict] = None
    data_root: Optional[str] = None
    results_root: Optional[str] = None
    device: Optional[str] = None
    seed: Optional[int] = None
    num_rounds: Optional[int] = None
    local_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None
    selected_attacks: Optional[List[str]] = None
    k_by_model: Dict[str, int] = field(default_factory=dict)
    model_dataset_map: Dict[str, str] = field(default_factory=lambda: {"lenet5": "mnist"})
    default_dataset_for_other_models: str = "cifar10"
    benign_reference_clients: Optional[int] = None
    max_proxy_samples: Optional[int] = None
    output_prefix: str = "layer_extraction"
    weighting_mode: str = "raw_variance_on_zscored_scores"
    epsilon: float = 1e-12
    print_progress: Optional[bool] = None
    save_text_log: Optional[bool] = None

    @classmethod
    def from_config(cls, base_config=Config, **overrides) -> "LayerExtractionSettings":
        k_by_model = overrides.pop("k_by_model", _default_k_by_model(base_config))
        return cls(base_config=base_config, k_by_model=k_by_model, **overrides)

    def get_model_name(self) -> str:
        value = self.model if self.model is not None else getattr(self.base_config, "MODEL")
        return str(value).strip().lower()

    def get_selected_models(self) -> List[str]:
        if self.selected_models is not None:
            normalized_models = [str(model_name).strip().lower() for model_name in self.selected_models]
            normalized_models = [model_name for model_name in normalized_models if model_name]
            if not normalized_models:
                raise ValueError("selected_models 不能为空列表。")
            return normalized_models
        return [self.get_model_name()]

    def resolve_dataset_for_model(self, model_name: Optional[str] = None) -> str:
        normalized_model = str(model_name or self.get_model_name()).strip().lower()
        # 批量模型模式下，数据集规则固定为 lenet5->mnist，其余模型->cifar10。
        if self.selected_models is not None:
            if normalized_model in self.model_dataset_map:
                return str(self.model_dataset_map[normalized_model]).strip().lower()
            return str(self.default_dataset_for_other_models).strip().lower()

        value = self.dataset if self.dataset is not None else getattr(self.base_config, "DATASET")
        return str(value).strip().lower()

    def get_dataset_name(self) -> str:
        return self.resolve_dataset_for_model(self.get_model_name())

    def get_partition_name(self) -> str:
        partition = self.partition if self.partition is not None else getattr(self.base_config, "PARTITION")
        if isinstance(partition, dict):
            return str(partition.get("type", "iid")).strip().lower()
        return str(partition).strip().lower()

    def get_partition_config(self) -> dict:
        partition = self.partition if self.partition is not None else getattr(self.base_config, "PARTITION")
        if hasattr(self.base_config, "_resolve_partition"):
            return dict(self.base_config._resolve_partition(partition))
        if isinstance(partition, dict):
            return dict(partition)
        return {"type": self.get_partition_name()}

    def get_experiment_name(self) -> str:
        if hasattr(self.base_config, "_format_experiment_name"):
            return self.base_config._format_experiment_name(
                model=self.get_model_name(),
                dataset=self.get_dataset_name(),
                partition_name=self.get_partition_name(),
            )
        return f"{self.get_model_name()}-{self.get_dataset_name()}-{self.get_partition_name()}"

    def get_data_root(self) -> str:
        return str(self.data_root if self.data_root is not None else getattr(self.base_config, "DATA_ROOT"))

    def get_results_root(self) -> str:
        return str(self.results_root if self.results_root is not None else getattr(self.base_config, "RESULTS_ROOT"))

    def get_seed(self) -> int:
        return int(self.seed if self.seed is not None else getattr(self.base_config, "SEED"))

    def get_num_rounds(self) -> int:
        return int(self.num_rounds if self.num_rounds is not None else getattr(self.base_config, "NUM_ROUNDS"))

    def get_local_epochs(self) -> int:
        return int(self.local_epochs if self.local_epochs is not None else getattr(self.base_config, "LOCAL_EPOCHS"))

    def get_batch_size(self) -> int:
        return int(self.batch_size if self.batch_size is not None else getattr(self.base_config, "BATCH_SIZE"))

    def get_learning_rate(self) -> float:
        return float(self.learning_rate if self.learning_rate is not None else getattr(self.base_config, "LR"))

    def get_momentum(self) -> float:
        return float(self.momentum if self.momentum is not None else getattr(self.base_config, "MOMENTUM"))

    def get_weight_decay(self) -> float:
        return float(self.weight_decay if self.weight_decay is not None else getattr(self.base_config, "WEIGHT_DECAY"))

    def get_device(self) -> str:
        if self.device is not None:
            if self.device == "cuda" and not torch.cuda.is_available():
                return "cpu"
            return str(self.device)
        if hasattr(self.base_config, "get_device"):
            return str(self.base_config.get_device())
        device = str(getattr(self.base_config, "DEVICE", "cpu"))
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device

    def get_num_clients(self) -> int:
        return int(getattr(self.base_config, "NUM_CLIENTS"))

    def get_num_malicious(self) -> int:
        if hasattr(self.base_config, "get_num_malicious"):
            resolved = int(self.base_config.get_num_malicious())
        else:
            poison_rate = float(getattr(self.base_config, "POISON_RATE", 0.0))
            resolved = int(self.get_num_clients() * poison_rate)
            if poison_rate > 0 and resolved == 0:
                resolved = 1
        # layer extraction 必然需要一个恶意分支，避免把工具跑成全良性空流程。
        return max(1, resolved)

    def get_benign_reference_clients(self) -> int:
        if self.benign_reference_clients is not None:
            return max(1, int(self.benign_reference_clients))
        return max(1, self.get_num_clients() - self.get_num_malicious())

    def get_attack_configs(self) -> List[Dict[str, object]]:
        selected_attacks = (
            self.selected_attacks
            if self.selected_attacks is not None
            else list(getattr(self.base_config, "SELECTED_ATTACKS"))
        )
        strengths = dict(getattr(self.base_config, "ATTACK_STRENGTHS", {}))
        strengths_by_dataset = dict(getattr(self.base_config, "ATTACK_STRENGTHS_BY_DATASET", {}))
        attack_configs: List[Dict[str, object]] = []
        for attack_name in selected_attacks:
            normalized_name = str(attack_name).strip().lower()
            attack_configs.append(
                {
                    "name": normalized_name,
                    "params": dict(strengths.get(normalized_name, {})),
                    "params_by_dataset": dict(strengths_by_dataset.get(normalized_name, {})),
                }
            )
        return attack_configs

    def get_k_for_model(self, model_name: Optional[str] = None) -> int:
        normalized_name = str(model_name or self.get_model_name()).strip().lower()
        if normalized_name not in self.k_by_model:
            raise KeyError(f"K_BY_MODEL 缺少模型 {normalized_name} 的预算配置。")
        return int(self.k_by_model[normalized_name])

    def for_model(self, model_name: str) -> "LayerExtractionSettings":
        normalized_model = str(model_name).strip().lower()
        return replace(
            self,
            model=normalized_model,
            dataset=self.resolve_dataset_for_model(normalized_model),
            selected_models=None,
        )

    def get_run_settings(self) -> List["LayerExtractionSettings"]:
        return [self.for_model(model_name) for model_name in self.get_selected_models()]

    def should_print_progress(self) -> bool:
        if self.print_progress is not None:
            return bool(self.print_progress)
        return bool(getattr(self.base_config, "PRINT_PROGRESS", True))

    def should_save_text_log(self) -> bool:
        if self.save_text_log is not None:
            return bool(self.save_text_log)
        return bool(getattr(self.base_config, "SAVE_TEXT_LOG", True))
