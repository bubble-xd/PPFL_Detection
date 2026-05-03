from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import TensorDataset

from data import build_federated_datasets, materialize_dataset
from models import build_model
from utils.logger import RunLogger
from utils.random import set_global_seed

from .candidates import get_candidate_layer_prefixes
from .export import create_output_dir, export_selection_artifacts
from .selection import select_layers
from .settings import LayerExtractionSettings
from .simulator import simulate_attack_summary


def _build_reference_model(settings, dataset_info: Dict[str, object]):
    return build_model(
        model_name=settings.get_model_name(),
        input_channels=dataset_info["input_channels"],
        num_classes=dataset_info["num_classes"],
        image_size=dataset_info["image_size"],
    )


def _build_proxy_dataset(settings, data_bundle) -> TensorDataset:
    proxy_dataset = materialize_dataset(data_bundle.train_dataset)
    if settings.max_proxy_samples is None:
        return proxy_dataset

    max_samples = int(settings.max_proxy_samples)
    if max_samples <= 0:
        raise ValueError("max_proxy_samples 必须为正数。")

    images, labels = proxy_dataset.tensors
    if max_samples >= len(labels):
        return proxy_dataset

    # 默认仍以全量干净训练集为入口；这里只给测试/调试保留可控降采样开关。
    generator = torch.Generator(device="cpu")
    generator.manual_seed(settings.get_seed())
    indices = torch.randperm(len(labels), generator=generator)[:max_samples]
    return TensorDataset(images[indices].clone(), labels[indices].clone())


def _run_single_model_extraction(
    settings: LayerExtractionSettings,
) -> Dict[str, object]:
    set_global_seed(settings.get_seed())

    output_dir = create_output_dir(
        results_root=settings.get_results_root(),
        output_prefix=settings.output_prefix,
        model_name=settings.get_model_name(),
        dataset_name=settings.get_dataset_name(),
        partition_name=settings.get_partition_name(),
    )
    logger = RunLogger(
        log_path=f"{output_dir}/run.log",
        print_to_console=settings.should_print_progress(),
    )
    logger.info(f"[Output] results_dir={output_dir}")

    data_bundle = build_federated_datasets(
        dataset_name=settings.get_dataset_name(),
        num_clients=settings.get_num_clients(),
        batch_size=settings.get_batch_size(),
        partition=settings.get_partition_config(),
        root=settings.get_data_root(),
    )
    proxy_dataset = _build_proxy_dataset(settings=settings, data_bundle=data_bundle)
    reference_model = _build_reference_model(settings=settings, dataset_info=data_bundle.dataset_info)
    candidate_layers = get_candidate_layer_prefixes(reference_model)
    logger.info(
        f"[Setup] experiment={settings.get_experiment_name()} candidates={candidate_layers} "
        f"proxy_size={len(proxy_dataset)}"
    )

    attack_summaries = []
    for attack_config in settings.get_attack_configs():
        attack_summary = simulate_attack_summary(
            settings=settings,
            dataset_info=data_bundle.dataset_info,
            proxy_dataset=proxy_dataset,
            candidate_layers=candidate_layers,
            attack_config=attack_config,
            experiment_name=settings.get_experiment_name(),
            logger=logger,
        )
        attack_summaries.append(attack_summary)

    selection_result = select_layers(
        model_name=settings.get_model_name(),
        dataset_name=settings.get_dataset_name(),
        partition_name=settings.get_partition_name(),
        num_rounds=settings.get_num_rounds(),
        candidate_layers=candidate_layers,
        attack_summaries=attack_summaries,
        k=settings.get_k_for_model(),
        weighting_mode=settings.weighting_mode,
    )
    exported_paths = export_selection_artifacts(
        output_dir=output_dir,
        result=selection_result,
        attack_summaries=attack_summaries,
    )
    logger.info(
        f"[Selection] k={selection_result.k} selected_layers={selection_result.selected_layers}"
    )
    logger.info(f"[Export] selection={exported_paths['selection_path']}")
    logger.info(f"[Export] layer_scores={exported_paths['layer_scores_path']}")

    return {
        "model": settings.get_model_name(),
        "dataset": settings.get_dataset_name(),
        "partition": settings.get_partition_name(),
        "output_dir": output_dir,
        "selection_path": exported_paths["selection_path"],
        "layer_scores_path": exported_paths["layer_scores_path"],
        "selected_layers": selection_result.selected_layers,
    }


def run_layer_extraction(
    settings: Optional[LayerExtractionSettings] = None,
) -> Dict[str, object]:
    settings = settings or LayerExtractionSettings.from_config()
    run_settings = settings.get_run_settings()

    run_results: List[Dict[str, object]] = [
        _run_single_model_extraction(model_settings)
        for model_settings in run_settings
    ]
    if len(run_results) == 1:
        return run_results[0]

    return {
        "runs": run_results,
        "output_dirs": [run_result["output_dir"] for run_result in run_results],
        "selected_layers_by_model": {
            str(run_result["model"]): list(run_result["selected_layers"])
            for run_result in run_results
        },
    }
