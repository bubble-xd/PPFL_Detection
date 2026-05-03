from __future__ import annotations

from typing import Dict, Sequence

import torch

from attacks import build_attack_adapter
from models import build_model
from utils.random import derive_seed, set_global_seed
from utils.state_dict import build_state_delta_dict, clone_state_dict
from utils.training import train_local_model

from .scoring import compute_round_layer_metrics, summarize_attack_scores
from .types import AttackLayerSummary


def _build_model_for_extraction(
    model_name: str,
    dataset_info: Dict[str, object],
):
    return build_model(
        model_name=model_name,
        input_channels=dataset_info["input_channels"],
        num_classes=dataset_info["num_classes"],
        image_size=dataset_info["image_size"],
    )


def _train_branch_state(
    settings,
    dataset_info: Dict[str, object],
    global_state: Dict[str, torch.Tensor],
    local_dataset,
    local_seed: int,
) -> Dict[str, torch.Tensor]:
    set_global_seed(local_seed)
    local_model = _build_model_for_extraction(
        model_name=settings.get_model_name(),
        dataset_info=dataset_info,
    )
    local_model.load_state_dict(global_state)
    trained_model = train_local_model(
        model=local_model,
        dataset=local_dataset,
        device=settings.get_device(),
        local_epochs=settings.get_local_epochs(),
        batch_size=settings.get_batch_size(),
        learning_rate=settings.get_learning_rate(),
        momentum=settings.get_momentum(),
        weight_decay=settings.get_weight_decay(),
        seed=local_seed,
    )
    return clone_state_dict(trained_model.state_dict(), device="cpu")


def simulate_attack_summary(
    settings,
    dataset_info: Dict[str, object],
    proxy_dataset,
    candidate_layers: Sequence[str],
    attack_config: Dict[str, object],
    experiment_name: str,
    logger=None,
) -> AttackLayerSummary:
    attack_name = str(attack_config["name"]).strip().lower()
    attack_adapter = build_attack_adapter(
        attack_config=attack_config,
        dataset_name=settings.get_dataset_name(),
        dataset_info=dataset_info,
    )
    malicious_ids = list(range(settings.get_num_malicious()))
    malicious_client_id = malicious_ids[0]
    benign_reference_count = (
        settings.get_benign_reference_clients()
        if attack_name in {"a_lie", "fedimp"}
        else 1
    )

    global_model = _build_model_for_extraction(
        model_name=settings.get_model_name(),
        dataset_info=dataset_info,
    )
    global_state = clone_state_dict(global_model.state_dict(), device="cpu")

    round_metrics = []
    if logger is not None:
        logger.info(
            f"[AttackStart] attack={attack_name} rounds={settings.get_num_rounds()} "
            f"benign_refs={benign_reference_count}"
        )

    for round_index in range(1, settings.get_num_rounds() + 1):
        round_seed = derive_seed(settings.get_seed(), experiment_name, attack_name, round_index)
        benign_states = []
        # 对 ALIE / FedImp 这类依赖良性群体统计的攻击，这里显式保留一个 benign pool。
        for benign_index in range(benign_reference_count):
            benign_seed = derive_seed(round_seed, "benign", benign_index)
            benign_state = _train_branch_state(
                settings=settings,
                dataset_info=dataset_info,
                global_state=global_state,
                local_dataset=proxy_dataset,
                local_seed=benign_seed,
            )
            benign_states.append(benign_state)

        benign_state = benign_states[0]
        malicious_dataset = proxy_dataset
        if attack_adapter.is_data_poisoning:
            malicious_dataset = attack_adapter.poison_local_dataset(
                dataset=proxy_dataset,
                client_id=malicious_client_id,
                malicious_ids=malicious_ids,
            )

        malicious_seed = derive_seed(round_seed, "malicious")
        malicious_state = _train_branch_state(
            settings=settings,
            dataset_info=dataset_info,
            global_state=global_state,
            local_dataset=malicious_dataset,
            local_seed=malicious_seed,
        )
        if attack_adapter.is_update_poisoning:
            malicious_state = attack_adapter.poison_client_state(
                current_state=malicious_state,
                client_id=malicious_client_id,
                malicious_ids=malicious_ids,
                benign_states=benign_states,
                global_state_dict=global_state,
                num_clients=settings.get_num_clients(),
            )

        benign_delta = build_state_delta_dict(benign_state, global_state)
        malicious_delta = build_state_delta_dict(malicious_state, global_state)
        metrics = compute_round_layer_metrics(
            candidate_layers=candidate_layers,
            benign_delta=benign_delta,
            malicious_delta=malicious_delta,
            round_index=round_index,
            epsilon=settings.epsilon,
        )
        round_metrics.append(metrics)
        if logger is not None:
            logger.info(
                f"[AttackRound] attack={attack_name} round={round_index}/{settings.get_num_rounds()} "
                f"alpha={metrics.alpha:.4f} beta={metrics.beta:.4f}"
            )

        # 下一轮全局模型沿良性轨迹推进，避免把攻击轨迹反向污染到控制基线。
        global_state = clone_state_dict(benign_state, device="cpu")

    summary = summarize_attack_scores(
        attack_name=attack_name,
        round_metrics=round_metrics,
        candidate_layers=candidate_layers,
    )
    if logger is not None:
        logger.info(
            f"[AttackDone] attack={attack_name} top1={summary.top1_layer} "
            f"score={summary.top1_score:.6f}"
        )
    return summary
