from __future__ import annotations

import difflib
from typing import Dict, List

from .adapters import ATTACK_ADAPTERS


def _resolve_attack_params(attack_config: Dict[str, object], dataset_name: str) -> Dict[str, object]:
    params = dict(attack_config.get("params", {}))
    params_by_dataset = dict(attack_config.get("params_by_dataset", {}))
    dataset_overrides = dict(params_by_dataset.get(dataset_name, {}))
    params.update(dataset_overrides)
    return params


def get_enabled_attack_configs(attack_configs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [attack_cfg for attack_cfg in attack_configs if attack_cfg.get("enabled", True)]


def build_attack_adapter(
    attack_config: Dict[str, object],
    dataset_name: str,
    dataset_info: Dict[str, object],
):
    attack_name = str(attack_config["name"]).strip().lower()
    params = _resolve_attack_params(attack_config, dataset_name)

    if attack_name not in ATTACK_ADAPTERS:
        supported = sorted(ATTACK_ADAPTERS.keys())
        suggestions = difflib.get_close_matches(attack_name, supported, n=3, cutoff=0.45)
        matched_subnames = [name for name in supported if name in attack_name and name != attack_name]

        details = []
        if matched_subnames:
            details.append(
                "possible missing comma between attack names: "
                + ", ".join(matched_subnames)
            )
        if suggestions:
            details.append("did you mean: " + ", ".join(suggestions))

        suffix = f" ({'; '.join(details)})" if details else ""
        raise ValueError(f"Unsupported attack: {attack_name}{suffix}")

    adapter_cls = ATTACK_ADAPTERS[attack_name]
    if attack_name == "label_flipping_untargeted":
        return adapter_cls(num_classes=dataset_info["num_classes"], params=params)
    if attack_name == "label_flipping_targeted":
        return adapter_cls(num_classes=dataset_info["num_classes"], params=params)
    if attack_name in {"badnets", "dba", "edge_case", "semantic_backdoor"}:
        return adapter_cls(dataset_name=dataset_name, params=params)
    return adapter_cls(params=params)
