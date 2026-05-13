from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List

import torch

from aggregators import run_aggregator
from attacks import build_attack_adapter
from data import build_federated_datasets
from features import FeatureBuilder
from metrics import compute_detection_f1, evaluate_accuracy, evaluate_asr
from models import build_model
from utils.io import create_poison_rate_directory, create_run_directory, export_experiment_results
from utils.heatmaps import (
    build_fixed_client_order,
    compute_cosine_group_metrics,
    pairwise_cosine_similarity,
    resolve_cosine_heatmap_rounds,
    save_cosine_heatmaps,
)
from utils.logger import RunLogger
from utils.random import derive_seed, set_global_seed
from utils.state_dict import (
    clone_state_dict,
)
from utils.state_store import DiskStateStore, LazyStateDeltaSequence
from utils.training import configure_torch_runtime, train_local_model


def _log_progress(config, logger: RunLogger, message: str) -> None:
    if getattr(config, "SAVE_TEXT_LOG", True) or getattr(config, "PRINT_PROGRESS", True):
        logger.info(message)


def _resolve_runtime_options(config, device: str) -> Dict[str, object]:
    raw_pin_memory = getattr(config, "PIN_MEMORY", None)
    pin_memory = (
        bool(raw_pin_memory)
        if raw_pin_memory is not None
        else str(device).startswith("cuda") and torch.cuda.is_available()
    )
    return {
        "num_workers": int(getattr(config, "DATA_LOADER_NUM_WORKERS", 0)),
        "pin_memory": pin_memory,
        "persistent_workers": bool(getattr(config, "DATA_LOADER_PERSISTENT_WORKERS", False)),
        "prefetch_factor": getattr(config, "DATA_LOADER_PREFETCH_FACTOR", None),
        "use_amp": bool(getattr(config, "USE_AMP", True)) and str(device).startswith("cuda") and torch.cuda.is_available(),
        "channels_last": bool(getattr(config, "USE_CHANNELS_LAST", True))
        and str(device).startswith("cuda")
        and torch.cuda.is_available(),
        "max_batches": getattr(config, "MAX_BATCHES", None),
    }


def _resolve_num_malicious(num_clients: int, poison_rate: float) -> int:
    num_malicious = int(num_clients * poison_rate)
    if poison_rate > 0 and num_malicious == 0:
        num_malicious = 1
    return min(num_clients, max(0, num_malicious))


def _select_malicious_clients(num_clients: int, poison_rate: float, seed: int) -> List[int]:
    num_malicious = _resolve_num_malicious(num_clients, poison_rate)
    if num_malicious == 0:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_clients), num_malicious))


def _json_list(values: List[int]) -> str:
    return json.dumps([int(value) for value in values])


def _build_model_from_experiment(config, experiment_cfg, dataset_info):
    return build_model(
        model_name=experiment_cfg["model"],
        input_channels=dataset_info["input_channels"],
        num_classes=dataset_info["num_classes"],
        image_size=dataset_info["image_size"],
    )


def _should_use_disk_state_store(config, model_name: str) -> bool:
    if hasattr(config, "should_stream_client_states"):
        return bool(config.should_stream_client_states(model_name))

    enabled_models = {
        str(enabled_model).strip().lower()
        for enabled_model in getattr(config, "DISK_STATE_CACHE_MODELS", ["vgg11"])
    }
    return str(model_name).strip().lower() in enabled_models


def _resolve_fedavg_feature_mode(feature_modes: List[str]) -> str:
    # FedAvg 不依赖特征提取；这里固定收敛到单一展示口径，
    normalized_feature_modes = [str(feature_mode).strip() for feature_mode in feature_modes if str(feature_mode).strip()]
    for feature_mode in normalized_feature_modes:
        if feature_mode.lower() == "raw_full":
            return feature_mode
    if normalized_feature_modes:
        return normalized_feature_modes[0]
    return "raw_full"


def _resolve_robust_method_feature_runs(
    methods: List[str],
    feature_modes: List[str],
) -> List[tuple[str, str]]:
    # 常规鲁棒聚合器需要扫描全部特征模式；但 FedAvg 与特征无关，因此每种攻击只保留一组 `(fedavg, raw_full)`。
    resolved_runs: List[tuple[str, str]] = []
    seen_runs = set()
    fedavg_feature_mode = _resolve_fedavg_feature_mode(feature_modes=feature_modes)

    for method_name in methods:
        normalized_method_name = str(method_name).strip()
        if not normalized_method_name:
            continue

        candidate_feature_modes = (
            [fedavg_feature_mode]
            if normalized_method_name.lower() == "fedavg"
            else [str(feature_mode).strip() for feature_mode in feature_modes if str(feature_mode).strip()]
        )
        if not candidate_feature_modes:
            candidate_feature_modes = [fedavg_feature_mode]

        for feature_mode in candidate_feature_modes:
            run_key = (normalized_method_name.lower(), feature_mode.lower())
            if run_key in seen_runs:
                continue
            seen_runs.add(run_key)
            resolved_runs.append((normalized_method_name, feature_mode))

    return resolved_runs


def _run_single_setting(
    config,
    logger: RunLogger,
    experiment_cfg: Dict[str, object],
    data_bundle,
    run_output_dir: str,
    feature_builder: FeatureBuilder,
    attack_name: str,
    attack_mode: str,
    malicious_ids: List[int],
    method_name: str,
    feature_mode: str,
    run_type: str,
    train_attack_adapter,
    eval_attack_adapter,
    asr_loader=None,
):
    device = config.get_device()
    runtime_options = _resolve_runtime_options(config, device)
    dataset_info = data_bundle.dataset_info
    set_global_seed(config.SEED)
    use_disk_state_store = _should_use_disk_state_store(config, experiment_cfg["model"])

    global_model = _build_model_from_experiment(config, experiment_cfg, dataset_info)
    global_state = clone_state_dict(global_model.state_dict(), device="cpu")
    if (
        asr_loader is None
        and eval_attack_adapter is not None
        and eval_attack_adapter.attack_mode == "targeted"
    ):
        asr_loader = eval_attack_adapter.build_asr_eval_loader(
            clean_test_dataset=data_bundle.test_dataset,
            batch_size=config.BATCH_SIZE,
        )

    _log_progress(
        config,
        logger,
        (
            f"[Start] experiment={experiment_cfg['name']} "
            f"attack={attack_name} run_type={run_type} "
            f"method={method_name} feature={feature_mode}"
        ),
    )
    round_logs = []
    heatmap_rounds = (
        set(resolve_cosine_heatmap_rounds(config.NUM_ROUNDS, getattr(config, "COSINE_HEATMAP_ROUNDS", None)))
        if getattr(config, "SAVE_COSINE_HEATMAPS", False)
        else set()
    )
    heatmap_anchor_method = (
        str(config.ROBUST_METHODS[0]).strip().lower() if getattr(config, "ROBUST_METHODS", None) else "fedavg"
    )
    heatmap_anchor_feature = (
        str(config.FEATURE_MODES[0]).strip().lower() if getattr(config, "FEATURE_MODES", None) else "raw_full"
    )
    should_export_heatmaps_for_setting = (
        bool(getattr(config, "SAVE_COSINE_HEATMAPS", False))
        and str(run_type).strip().lower() == "robust"
        and str(method_name).strip().lower() == heatmap_anchor_method
        and str(feature_mode).strip().lower() == heatmap_anchor_feature
    )

    def _aggregate_round(local_state_dicts):
        if method_name == "fedavg":
            aggregation_result = run_aggregator(
                method_name="fedavg",
                local_state_dicts=local_state_dicts,
                feature_matrix=None,
                num_malicious=len(malicious_ids),
                global_state_dict=global_state,
                config=config,
            )
            predicted_malicious_ids: List[int] = []
            bm_gap = float("nan")
            return aggregation_result, predicted_malicious_ids, bm_gap

        # BM-Gap 与离线选层都关注“客户端更新”的可分性；
        # 这里改为在 delta 空间构造特征，避免共享全局权重把余弦相似度整体抬到接近 1。
        feature_state_dicts = LazyStateDeltaSequence(
            state_dicts=local_state_dicts,
            global_state_dict=global_state,
        )
        feature_set = feature_builder.build_feature_set(
            local_state_dicts=feature_state_dicts,
            feature_mode=feature_mode,
        )
        aggregation_result = run_aggregator(
            method_name=method_name,
            local_state_dicts=local_state_dicts,
            feature_matrix=feature_set.aggregator_matrix,
            num_malicious=len(malicious_ids),
            global_state_dict=global_state,
            config=config,
        )
        predicted_malicious_ids = aggregation_result.predicted_malicious_ids
        # BM-Gap 直接基于当前轮特征矩阵在内存中计算，
        # 不依赖热力图文件落盘，便于在关闭图像导出时仍保留该指标。
        similarity_matrix = (
            feature_set.cosine_similarity_matrix
            if feature_set.cosine_similarity_matrix is not None
            else pairwise_cosine_similarity(feature_set.aggregator_matrix)
        )
        cosine_metrics = compute_cosine_group_metrics(
            similarity_matrix=similarity_matrix,
            client_order=list(range(len(local_state_dicts))),
            malicious_ids=malicious_ids,
        )
        bm_gap = float(cosine_metrics["bm_gap"])
        return aggregation_result, predicted_malicious_ids, bm_gap

    def _maybe_save_round_heatmaps(local_state_dicts, round_idx: int) -> None:
        if not should_export_heatmaps_for_setting or int(round_idx) not in heatmap_rounds:
            return

        # 热力图这里按“客户端最终上传的完整本地模型”来画，
        # 显式保留共享的全局模型底座，方便后续做“提取后更显著”的视觉对比。
        heatmap_feature_matrices = {}
        for exported_feature_mode in getattr(config, "FEATURE_MODES", []):
            heatmap_feature_set = feature_builder.build_feature_set(
                local_state_dicts=local_state_dicts,
                feature_mode=exported_feature_mode,
            )
            heatmap_feature_matrices[exported_feature_mode] = heatmap_feature_set.aggregator_matrix

        saved_paths = save_cosine_heatmaps(
            output_dir=run_output_dir,
            experiment_name=experiment_cfg["name"],
            attack_name=attack_name,
            attack_mode=attack_mode,
            round_idx=int(round_idx),
            feature_matrices=heatmap_feature_matrices,
            client_order=build_fixed_client_order(config.NUM_CLIENTS, malicious_ids),
            malicious_ids=malicious_ids,
            feature_display_names=config.FEATURE_DISPLAY_NAMES,
            artifact_subdir="shared_global_base",
            similarity_space_tag="shared_global_base_local_models",
            similarity_space_description="Cosine similarity on local client models with shared global base",
        )
        _log_progress(
            config,
            logger,
            (
                f"[Heatmap] experiment={experiment_cfg['name']} attack={attack_name} "
                f"round={int(round_idx)} saved={saved_paths[0]}"
            ),
        )

    for round_idx in range(1, config.NUM_ROUNDS + 1):
        _log_progress(
            config,
            logger,
            (
                f"[Round {round_idx}/{config.NUM_ROUNDS}] "
                f"experiment={experiment_cfg['name']} "
                f"attack={attack_name} run_type={run_type} "
                f"method={method_name} feature={feature_mode}"
            ),
        )
        if use_disk_state_store:
            with DiskStateStore.create_temporary(
                parent_dir=run_output_dir,
                prefix=f".round_{int(round_idx):03d}_state_cache_",
            ) as round_state_store:
                # 同一轮内复用一个本地模型对象，每个客户端训练前重新加载全局参数。
                # 这样避免 ResNet/VGG 在每个客户端上重复构造网络模块。
                local_model = _build_model_from_experiment(config, experiment_cfg, dataset_info)
                for client_id in range(config.NUM_CLIENTS):
                    local_seed = derive_seed(config.SEED, experiment_cfg["name"], attack_name, round_idx, client_id)
                    set_global_seed(local_seed)

                    local_dataset = data_bundle.client_datasets[client_id]
                    if (
                        train_attack_adapter is not None
                        and train_attack_adapter.is_data_poisoning
                        and client_id in malicious_ids
                    ):
                        local_dataset = train_attack_adapter.poison_local_dataset(
                            dataset=local_dataset,
                            client_id=client_id,
                            malicious_ids=malicious_ids,
                        )

                    local_model.load_state_dict(global_state)
                    local_model = train_local_model(
                        model=local_model,
                        dataset=local_dataset,
                        device=device,
                        local_epochs=config.LOCAL_EPOCHS,
                        batch_size=config.BATCH_SIZE,
                        learning_rate=config.LR,
                        momentum=config.MOMENTUM,
                        weight_decay=config.WEIGHT_DECAY,
                        seed=local_seed,
                        num_workers=runtime_options["num_workers"],
                        pin_memory=runtime_options["pin_memory"],
                        persistent_workers=runtime_options["persistent_workers"],
                        prefetch_factor=runtime_options["prefetch_factor"],
                        use_amp=runtime_options["use_amp"],
                        channels_last=runtime_options["channels_last"],
                        max_batches=runtime_options["max_batches"],
                    )
                    local_state = clone_state_dict(local_model.state_dict(), device="cpu")
                    round_state_store.save_state(client_id=client_id, state_dict=local_state)

                    # 只有超大模型才默认走磁盘缓存路径，降低 round 内完整 state 的常驻峰值。
                    del local_state
                del local_model

                if train_attack_adapter is not None and train_attack_adapter.is_update_poisoning and malicious_ids:
                    benign_ids = [
                        client_id
                        for client_id in round_state_store.get_client_ids()
                        if client_id not in malicious_ids
                    ]
                    benign_states = round_state_store.build_view(benign_ids)
                    for client_id in malicious_ids:
                        current_state = round_state_store.load_state(client_id)
                        # 更新投毒攻击内部仍可在 update 空间构造攻击，
                        # 但 round 内只在需要时回读当前恶意客户端和良性视图。
                        poisoned_state = train_attack_adapter.poison_client_state(
                            current_state=current_state,
                            client_id=client_id,
                            malicious_ids=malicious_ids,
                            benign_states=benign_states,
                            global_state_dict=global_state,
                            num_clients=config.NUM_CLIENTS,
                        )
                        round_state_store.save_state(client_id=client_id, state_dict=poisoned_state)

                round_state_view = round_state_store.build_view()
                _maybe_save_round_heatmaps(round_state_view, round_idx=round_idx)
                aggregation_result, predicted_malicious_ids, bm_gap = _aggregate_round(
                    round_state_view
                )
        else:
            client_states = []
            # 普通内存路径也复用模型实例；客户端隔离由每次 load_state_dict(global_state) 保证。
            local_model = _build_model_from_experiment(config, experiment_cfg, dataset_info)
            for client_id in range(config.NUM_CLIENTS):
                local_seed = derive_seed(config.SEED, experiment_cfg["name"], attack_name, round_idx, client_id)
                set_global_seed(local_seed)

                local_dataset = data_bundle.client_datasets[client_id]
                if (
                    train_attack_adapter is not None
                    and train_attack_adapter.is_data_poisoning
                    and client_id in malicious_ids
                ):
                    local_dataset = train_attack_adapter.poison_local_dataset(
                        dataset=local_dataset,
                        client_id=client_id,
                        malicious_ids=malicious_ids,
                    )

                local_model.load_state_dict(global_state)
                local_model = train_local_model(
                    model=local_model,
                    dataset=local_dataset,
                    device=device,
                    local_epochs=config.LOCAL_EPOCHS,
                    batch_size=config.BATCH_SIZE,
                    learning_rate=config.LR,
                    momentum=config.MOMENTUM,
                    weight_decay=config.WEIGHT_DECAY,
                    seed=local_seed,
                    num_workers=runtime_options["num_workers"],
                    pin_memory=runtime_options["pin_memory"],
                    persistent_workers=runtime_options["persistent_workers"],
                    prefetch_factor=runtime_options["prefetch_factor"],
                    use_amp=runtime_options["use_amp"],
                    channels_last=runtime_options["channels_last"],
                    max_batches=runtime_options["max_batches"],
                )
                client_states.append(
                    {
                        "client_id": client_id,
                        "state": clone_state_dict(local_model.state_dict(), device="cpu"),
                    }
                )
            del local_model

            if train_attack_adapter is not None and train_attack_adapter.is_update_poisoning and malicious_ids:
                benign_states = [
                    entry["state"]
                    for entry in client_states
                    if entry["client_id"] not in malicious_ids
                ]
                for entry in client_states:
                    if entry["client_id"] in malicious_ids:
                        entry["state"] = train_attack_adapter.poison_client_state(
                            current_state=entry["state"],
                            client_id=entry["client_id"],
                            malicious_ids=malicious_ids,
                            benign_states=benign_states,
                            global_state_dict=global_state,
                            num_clients=config.NUM_CLIENTS,
                        )
            sorted_local_states = [entry["state"] for entry in sorted(client_states, key=lambda item: item["client_id"])]
            _maybe_save_round_heatmaps(sorted_local_states, round_idx=round_idx)
            aggregation_result, predicted_malicious_ids, bm_gap = _aggregate_round(
                sorted_local_states
            )

        # 聚合器现在直接返回新的全局模型参数，而不是“聚合后的 update”。
        global_state = clone_state_dict(aggregation_result.aggregated_state, device="cpu")
        global_model.load_state_dict(global_state)
        acc = evaluate_accuracy(
            global_model,
            data_bundle.test_loader,
            device=device,
            use_amp=runtime_options["use_amp"],
            channels_last=runtime_options["channels_last"],
            non_blocking=runtime_options["pin_memory"],
        )
        asr = evaluate_asr(
            global_model,
            asr_loader,
            device=device,
            use_amp=runtime_options["use_amp"],
            channels_last=runtime_options["channels_last"],
            non_blocking=runtime_options["pin_memory"],
        )
        f1 = (
            float("nan")
            if method_name == "fedavg"
            else compute_detection_f1(predicted_malicious_ids, malicious_ids, config.NUM_CLIENTS)
        )
        round_logs.append(
            {
                "experiment_name": experiment_cfg["name"],
                "model": experiment_cfg["model"],
                "dataset": experiment_cfg["dataset"],
                "partition_type": data_bundle.partition_info["type"],
                "dirichlet_alpha": data_bundle.partition_info["alpha"],
                "attack_name": attack_name,
                "attack_mode": attack_mode,
                "run_type": run_type,
                "method": method_name,
                "feature_mode": feature_mode,
                "round": round_idx,
                "f1": f1,
                "acc": acc,
                "asr": asr,
                "bm_gap": bm_gap,
                "malicious_client_ids": _json_list(malicious_ids),
                "predicted_malicious_ids": _json_list(predicted_malicious_ids if method_name != "fedavg" else []),
                "selected_client_ids": _json_list(aggregation_result.selected_client_ids),
            }
        )
        f1_repr = "nan" if math.isnan(float(f1)) else f"{float(f1):.4f}"
        asr_repr = "nan" if math.isnan(float(asr)) else f"{float(asr):.4f}"
        bm_gap_repr = "nan" if math.isnan(float(bm_gap)) else f"{float(bm_gap):.4f}"
        _log_progress(
            config,
            logger,
            (
                f"[Detect] experiment={experiment_cfg['name']} "
                f"attack={attack_name} run_type={run_type} "
                f"method={method_name} feature={feature_mode} "
                f"round={round_idx}/{config.NUM_ROUNDS} "
                f"true_malicious={malicious_ids} "
                f"predicted={predicted_malicious_ids if method_name != 'fedavg' else []} "
                f"selected={aggregation_result.selected_client_ids}"
            ),
        )
        _log_progress(
            config,
            logger,
            (
                f"[Metrics] experiment={experiment_cfg['name']} "
                f"attack={attack_name} run_type={run_type} "
                f"method={method_name} feature={feature_mode} "
                f"round={round_idx}/{config.NUM_ROUNDS} "
                f"acc={float(acc):.4f} asr={asr_repr} f1={f1_repr} bm_gap={bm_gap_repr}"
            ),
        )

    _log_progress(
        config,
        logger,
        (
            f"[Done] experiment={experiment_cfg['name']} "
            f"attack={attack_name} run_type={run_type} "
            f"method={method_name} feature={feature_mode}"
        ),
    )
    return round_logs


def _summarize_round_logs(
    round_logs: List[Dict[str, object]],
    asr_tail_rounds: int = 1,
) -> Dict[str, float]:
    if not round_logs:
        raise ValueError("round_logs 不能为空。")

    def _safe_mean(key: str) -> float:
        values = [float(row[key]) for row in round_logs if not math.isnan(float(row[key]))]
        return float(sum(values) / len(values)) if values else float("nan")

    def _safe_tail_mean(key: str, tail_rounds: int) -> float:
        effective_tail_rounds = max(1, int(tail_rounds))
        tail_rows = round_logs[-effective_tail_rounds:]
        values = [
            float(row[key])
            for row in tail_rows
            if not math.isnan(float(row[key]))
        ]
        return float(sum(values) / len(values)) if values else float("nan")

    effective_asr_tail_rounds = max(1, int(asr_tail_rounds))
    final_row = round_logs[-1]
    tail_mean_asr = _safe_tail_mean("asr", effective_asr_tail_rounds)
    return {
        "mean_f1": _safe_mean("f1"),
        # BM-Gap 仅对使用特征矩阵的 robust 方案有意义；缺失时保持 NaN。
        "mean_bm_gap": _safe_mean("bm_gap"),
        # ACC 仍保留最后一轮；ASR 单轮波动更大，Excel 汇总改用最后 N 轮均值。
        "final_acc": float(final_row["acc"]),
        "final_asr": tail_mean_asr,
        "tail_mean_asr": tail_mean_asr,
        "last_asr": float(final_row["asr"]),
        "asr_tail_rounds": float(min(effective_asr_tail_rounds, len(round_logs))),
        "final_bm_gap": float(final_row.get("bm_gap", float("nan"))),
    }


def _build_summary_record(
    experiment_cfg: Dict[str, object],
    data_bundle,
    attack_name: str,
    attack_mode: str,
    run_type: str,
    method_name: str,
    feature_mode: str,
    summary: Dict[str, float],
    malicious_ids: List[int],
) -> Dict[str, object]:
    # 汇总记录按“攻击 + 方法 + 特征模式”保留结果，同时额外写入 run_type 便于排查。
    return {
        "experiment_name": experiment_cfg["name"],
        "model": experiment_cfg["model"],
        "dataset": experiment_cfg["dataset"],
        "partition_type": data_bundle.partition_info["type"],
        "dirichlet_alpha": data_bundle.partition_info["alpha"],
        "attack_name": attack_name,
        "attack_mode": attack_mode,
        "run_type": run_type,
        "method": method_name,
        "feature_mode": feature_mode,
        **summary,
        "malicious_client_ids": _json_list(malicious_ids),
    }


def _upsert_summary_record(
    summary_records: List[Dict[str, object]],
    summary_record: Dict[str, object],
) -> None:
    # 汇总表按“实验 + 攻击 + 方法 + 特征模式”唯一化，避免重复调度时写出重复记录。
    summary_key = (
        summary_record["experiment_name"],
        summary_record["attack_name"],
        summary_record["method"],
        summary_record["feature_mode"],
    )
    for index, existing_record in enumerate(summary_records):
        existing_key = (
            existing_record["experiment_name"],
            existing_record["attack_name"],
            existing_record["method"],
            existing_record["feature_mode"],
        )
        if existing_key == summary_key:
            summary_records[index] = summary_record
            return
    summary_records.append(summary_record)


def _run_all_experiments_for_active_poison_rate(
    config,
    run_output_dir: str,
) -> str:
    set_global_seed(config.SEED)
    os.makedirs(run_output_dir, exist_ok=True)
    log_path = f"{run_output_dir}/{config.TEXT_LOG_FILENAME}"
    logger = RunLogger(
        log_path=log_path,
        print_to_console=getattr(config, "PRINT_PROGRESS", True),
    )
    _log_progress(config, logger, f"[Output] results_dir={run_output_dir}")
    device = config.get_device()
    runtime_options = _resolve_runtime_options(config, device)
    _log_progress(
        config,
        logger,
        (
            f"[Runtime] device={device} cuda_available={torch.cuda.is_available()} "
            f"amp={runtime_options['use_amp']} channels_last={runtime_options['channels_last']} "
            f"num_workers={runtime_options['num_workers']} pin_memory={runtime_options['pin_memory']} "
            f"server_device={config.get_server_compute_device() if hasattr(config, 'get_server_compute_device') else device}"
        ),
    )
    _log_progress(
        config,
        logger,
        f"[PoisonRate] value={float(getattr(config, 'POISON_RATE', 0.0)):.4f}",
    )

    try:
        enabled_experiments = config.get_enabled_experiments()
        enabled_attack_configs = config.get_enabled_attack_configs()

        for experiment_cfg in enabled_experiments:
            _log_progress(
                config,
                logger,
                (
                    f"[Experiment] name={experiment_cfg['name']} "
                    f"model={experiment_cfg['model']} dataset={experiment_cfg['dataset']}"
                ),
            )
            partition_cfg = dict(experiment_cfg.get("partition", {"type": "iid"}))
            partition_cfg.setdefault("seed", derive_seed(config.SEED, experiment_cfg["name"], "partition"))
            if partition_cfg.get("type", "iid") == "dirichlet":
                partition_cfg.setdefault("alpha", config.DIRICHLET_ALPHA)

            data_bundle = build_federated_datasets(
                dataset_name=experiment_cfg["dataset"],
                num_clients=config.NUM_CLIENTS,
                batch_size=config.BATCH_SIZE,
                partition=partition_cfg,
                root=config.DATA_ROOT,
                num_workers=runtime_options["num_workers"],
                pin_memory=runtime_options["pin_memory"],
                persistent_workers=runtime_options["persistent_workers"],
                prefetch_factor=runtime_options["prefetch_factor"],
            )
            projection_seed = derive_seed(config.PROJECTION_SEED, experiment_cfg["name"])
            feature_builder = FeatureBuilder(
                model_name=experiment_cfg["model"],
                key_layer_map=config.KEY_LAYER_MAP,
                control_layer_map=config.CONTROL_LAYER_MAP,
                projection_dim=config.PROJECTION_DIM,
                projection_seed=projection_seed,
                feature_chunk_size=getattr(config, "FEATURE_STREAM_CHUNK_SIZE", 65536),
                max_dense_feature_bytes=int(getattr(config, "FEATURE_MATRIX_MAX_MB", 512) * 1024 * 1024),
                max_projection_matrix_bytes=int(
                    getattr(config, "PROJECTION_MATRIX_MAX_MB", 256) * 1024 * 1024
                ),
                compute_device=(
                    config.get_server_compute_device()
                    if hasattr(config, "get_server_compute_device")
                    else device
                ),
                max_gpu_feature_bytes=int(
                    getattr(config, "SERVER_FEATURE_GPU_MAX_MB", 512) * 1024 * 1024
                ),
                balanced_extra_layer_map=getattr(config, "BALANCED_EXTRA_LAYER_MAP", {}),
                include_batch_norm_in_balanced=bool(
                    getattr(config, "INCLUDE_BATCH_NORM_IN_BALANCED_FEATURES", False)
                ),
            )

            summary_records: List[Dict[str, object]] = []
            round_logs: List[Dict[str, object]] = []

            for attack_cfg in enabled_attack_configs:
                attack_adapter = build_attack_adapter(
                    attack_config=attack_cfg,
                    dataset_name=experiment_cfg["dataset"],
                    dataset_info=data_bundle.dataset_info,
                )
                malicious_seed = derive_seed(
                    config.MALICIOUS_CLIENT_SELECTION_SEED,
                    experiment_cfg["name"],
                    attack_cfg["name"],
                )
                malicious_ids = _select_malicious_clients(
                    num_clients=config.NUM_CLIENTS,
                    poison_rate=config.POISON_RATE,
                    seed=malicious_seed,
                )
                _log_progress(
                    config,
                    logger,
                    (
                        f"[Attack] experiment={experiment_cfg['name']} "
                        f"attack={attack_cfg['name']} mode={attack_adapter.attack_mode} "
                        f"malicious_clients={malicious_ids}"
                    ),
                )
                asr_loader = None
                if attack_adapter.attack_mode == "targeted":
                    # ASR 评估集只依赖攻击配置和测试集；
                    # 同一攻击下所有 method/feature 复用，避免重复构造投毒评估数据。
                    asr_loader = attack_adapter.build_asr_eval_loader(
                        clean_test_dataset=data_bundle.test_dataset,
                        batch_size=config.BATCH_SIZE,
                    )

                robust_runs = _resolve_robust_method_feature_runs(
                    methods=config.ROBUST_METHODS,
                    feature_modes=config.FEATURE_MODES,
                )
                for method_name, feature_mode in robust_runs:
                    robust_logs = _run_single_setting(
                        config=config,
                        logger=logger,
                        experiment_cfg=experiment_cfg,
                        data_bundle=data_bundle,
                        run_output_dir=run_output_dir,
                        feature_builder=feature_builder,
                        attack_name=attack_cfg["name"],
                        attack_mode=attack_adapter.attack_mode,
                        malicious_ids=malicious_ids,
                        method_name=method_name,
                        feature_mode=feature_mode,
                        run_type="robust",
                        train_attack_adapter=attack_adapter,
                        eval_attack_adapter=attack_adapter,
                        asr_loader=asr_loader,
                    )
                    round_logs.extend(robust_logs)
                    _upsert_summary_record(
                        summary_records,
                        _build_summary_record(
                            experiment_cfg=experiment_cfg,
                            data_bundle=data_bundle,
                            attack_name=attack_cfg["name"],
                            attack_mode=attack_adapter.attack_mode,
                            run_type="robust",
                            method_name=method_name,
                            feature_mode=feature_mode,
                            summary=_summarize_round_logs(
                                robust_logs,
                                asr_tail_rounds=getattr(config, "ASR_SUMMARY_TAIL_ROUNDS", 1),
                            ),
                            malicious_ids=malicious_ids,
                        ),
                    )

            export_experiment_results(
                output_dir=run_output_dir,
                experiment_name=experiment_cfg["name"],
                summary_records=summary_records,
                round_logs=round_logs,
                krum_score_logs=[],
                attacks=[attack_cfg["name"] for attack_cfg in enabled_attack_configs],
                methods=config.ROBUST_METHODS,
                feature_modes=config.FEATURE_MODES,
                method_display_names=config.METHOD_DISPLAY_NAMES,
                feature_display_names=config.FEATURE_DISPLAY_NAMES,
                poison_rate=config.POISON_RATE,
                save_csv_logs=config.SAVE_CSV_LOGS,
                save_round_logs=config.SAVE_ROUND_LOGS,
                export_excel=config.EXPORT_EXCEL,
                excel_metric_sheets=getattr(config, "EXCEL_METRIC_SHEETS", "all"),
            )
            _log_progress(
                config,
                logger,
                f"[Export] experiment={experiment_cfg['name']} files_saved_in={run_output_dir}",
            )
    except Exception:
        logger.exception("Benchmark execution failed.")
        raise

    return run_output_dir


def run_all_experiments(config) -> str:
    configure_torch_runtime(config)
    poison_rates = (
        config.get_poison_rates()
        if hasattr(config, "get_poison_rates")
        else [float(getattr(config, "POISON_RATE", 0.0))]
    )
    original_poison_rate = getattr(config, "POISON_RATE", None)
    base_output_dir = create_run_directory(config.RESULTS_ROOT, config=config)

    try:
        for poison_rate in poison_rates:
            # 当 `POISON_RATE` 配成列表时，这里逐个切换当前实验强度，
            # 保证每个投毒比例都落到独立结果目录，而不是混在同一份日志里。
            setattr(config, "POISON_RATE", float(poison_rate))
            poison_rate_output_dir = create_poison_rate_directory(
                parent_dir=base_output_dir,
                poison_rate=float(poison_rate),
            )
            _run_all_experiments_for_active_poison_rate(
                config,
                run_output_dir=poison_rate_output_dir,
            )
    finally:
        if original_poison_rate is not None:
            setattr(config, "POISON_RATE", original_poison_rate)

    return base_output_dir
