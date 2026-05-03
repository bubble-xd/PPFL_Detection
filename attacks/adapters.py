from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from attacks.targeted.badnets import BadNetsAttack
from attacks.targeted.dba import DBAAttack
from attacks.targeted.edge_case import EdgeCaseAttack
from attacks.targeted.label_flipping_targeted import label_flipping_targeted
from attacks.targeted.semantic_backdoor import SemanticBackdoorAttack
from attacks.untargeted.a_lie import alie_attack_update, build_alie_update_stats
from attacks.untargeted.additive_noise import additive_noise_attack
from attacks.untargeted.fedimp import build_fedimp_simulated_update_stats, fedimp_attack
from attacks.untargeted.label_flipping_untargeted import label_flipping_untargeted
from attacks.untargeted.random_gradient import random_gradient_attack
from attacks.untargeted.scaling_attack import scaling_attack
from attacks.untargeted.sign_flipping import sign_flipping
from data.data_loader import materialize_dataset
from utils.state_dict import build_state_delta_dict
from utils.state_store import LazyStateDeltaSequence


def _rebuild_state_from_delta(
    current_state_dict: Dict[str, torch.Tensor],
    global_state_dict: Dict[str, torch.Tensor],
    delta_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    # 客户端现在对外提交的是完整本地模型，因此这里要以当前本地模型为底板重建。
    # 这样只会覆盖攻击定义所在的浮点参数，不会把 BN 计数器等非浮点状态回滚成全局值。
    rebuilt_state = dict(current_state_dict)
    for key, global_value in global_state_dict.items():
        if (
            key in delta_dict
            and torch.is_tensor(global_value)
            and global_value.dtype.is_floating_point
        ):
            rebuilt_state[key] = global_value.detach().clone() + delta_dict[key].to(
                device=global_value.device,
                dtype=global_value.dtype,
            )
    return rebuilt_state


def _poison_local_state_via_update(
    current_state: Dict[str, torch.Tensor],
    global_state_dict: Dict[str, torch.Tensor],
    attack_fn,
    **attack_kwargs,
) -> Dict[str, torch.Tensor]:
    # 这些无目标攻击本质上定义在 update 空间里，
    # 但客户端对外提交的仍然应该是“投毒后的本地模型”。
    current_update = build_state_delta_dict(current_state, global_state_dict)
    poisoned_update = attack_fn(current_update, **attack_kwargs)
    return _rebuild_state_from_delta(current_state, global_state_dict, poisoned_update)


def _project_update_dict(
    update_dict: Dict[str, torch.Tensor],
    epsilon: Optional[float],
    projection_type: str,
) -> Dict[str, torch.Tensor]:
    """在 update 空间执行投影，避免恶意更新在放大前完全失控。"""
    projected_update = {key: value.clone() for key, value in update_dict.items()}
    if epsilon is None:
        return projected_update

    epsilon = float(epsilon)
    if epsilon <= 0:
        return projected_update

    normalized_projection = str(projection_type).strip().lower()
    if normalized_projection in {"l_2", "l2"}:
        # 这里统一按整体 L2 范数裁剪，保证不同层之间的相对方向不被破坏。
        if not projected_update:
            return projected_update
        # 流式累计整体范数，避免为了一个标量把所有层展平成超大临时向量。
        squared_norm = 0.0
        for tensor in projected_update.values():
            squared_norm += float(torch.sum(tensor.detach().to(dtype=torch.float32).pow(2)).item())
        global_norm = squared_norm ** 0.5
        if global_norm <= epsilon:
            return projected_update

        scale = epsilon / global_norm
        for key in projected_update:
            projected_update[key] = projected_update[key] * scale
        return projected_update

    if normalized_projection in {"l_inf", "linf"}:
        # L_inf 投影逐元素裁剪，更适合直接限制单参数最大偏移。
        for key in projected_update:
            projected_update[key] = torch.clamp(projected_update[key], -epsilon, epsilon)
        return projected_update

    raise ValueError(f"Unsupported projection_type: {projection_type}")


def _constrain_and_scale_local_state(
    current_state: Dict[str, torch.Tensor],
    global_state_dict: Dict[str, torch.Tensor],
    epsilon: Optional[float],
    projection_type: str,
    scaling_factor: float,
) -> Dict[str, torch.Tensor]:
    """
    先把恶意 update 约束到指定球内，再做 model replacement 风格的放大。
    这样既保留后门方向，又能显著提升恶意客户端对全局聚合的影响力。
    """
    current_update = build_state_delta_dict(current_state, global_state_dict)
    projected_update = _project_update_dict(
        current_update,
        epsilon=epsilon,
        projection_type=projection_type,
    )

    effective_scaling = float(scaling_factor)
    if effective_scaling <= 0:
        raise ValueError("scaling_factor 必须为正数。")

    if effective_scaling != 1.0:
        projected_update = scaling_attack(projected_update, gamma=effective_scaling)
    return _rebuild_state_from_delta(current_state, global_state_dict, projected_update)


def _resolve_dataset_tensors(dataset):
    # 已经物化过的客户端数据直接复用底层张量，投毒函数内部会 clone 被修改的数据。
    if isinstance(dataset, TensorDataset):
        return dataset.tensors
    return materialize_dataset(dataset).tensors


class BaseAttackAdapter:
    attack_name = "base"
    attack_mode = "untargeted"
    is_data_poisoning = False
    is_update_poisoning = False

    def poison_local_dataset(self, dataset, client_id: int, malicious_ids: List[int]):
        return materialize_dataset(dataset)

    def poison_client_state(
        self,
        current_state: Dict[str, torch.Tensor],
        client_id: int,
        malicious_ids: List[int],
        benign_states,
        global_state_dict,
        num_clients: int,
    ) -> Dict[str, torch.Tensor]:
        return current_state

    def build_asr_eval_loader(self, clean_test_dataset, batch_size: int):
        return None


class BadNetsAdapter(BaseAttackAdapter):
    attack_name = "badnets"
    attack_mode = "targeted"
    is_data_poisoning = True

    def __init__(self, dataset_name: str, params: Dict[str, object]) -> None:
        self.attack = BadNetsAttack(
            dataset_name=dataset_name,
            target_label=params.get("target_label", 0),
            poisoning_ratio=params.get("poisoning_ratio", 0.2),
            trigger_size=params.get("trigger_size", 3),
        )

    def poison_local_dataset(self, dataset, client_id: int, malicious_ids: List[int]):
        return self.attack.poison_dataset(dataset, train=True)

    def build_asr_eval_loader(self, clean_test_dataset, batch_size: int):
        return self.attack.get_poisoned_loader(
            clean_test_dataset,
            batch_size=batch_size,
            train=False,
            shuffle=False,
        )


class DBAAdapter(BaseAttackAdapter):
    attack_name = "dba"
    attack_mode = "targeted"
    is_data_poisoning = True

    def __init__(self, dataset_name: str, params: Dict[str, object]) -> None:
        self.dataset_name = dataset_name
        self.params = dict(params)

    def _build_attack(self, shard_id: int, num_shards: int) -> DBAAttack:
        return DBAAttack(
            dataset_name=self.dataset_name,
            target_label=self.params.get("target_label", 0),
            poisoning_ratio=self.params.get("poisoning_ratio", 0.2),
            trigger_size=self.params.get("trigger_size", 3),
            shard_id=shard_id,
            num_shards=max(1, num_shards),
        )

    def poison_local_dataset(self, dataset, client_id: int, malicious_ids: List[int]):
        shard_id = malicious_ids.index(client_id) if client_id in malicious_ids else 0
        attack = self._build_attack(shard_id=shard_id, num_shards=len(malicious_ids))
        return attack.poison_dataset(dataset, train=True)

    def build_asr_eval_loader(self, clean_test_dataset, batch_size: int):
        attack = self._build_attack(shard_id=0, num_shards=1)
        return attack.get_poisoned_loader(
            clean_test_dataset,
            batch_size=batch_size,
            train=False,
            shuffle=False,
        )


class EdgeCaseAdapter(BaseAttackAdapter):
    attack_name = "edge_case"
    attack_mode = "targeted"
    is_data_poisoning = True
    is_update_poisoning = True  # edge-case 需要 constrain-and-scale 才能形成更强后门

    def __init__(self, dataset_name: str, params: Dict[str, object]) -> None:
        self.attack = EdgeCaseAttack(
            dataset_name=dataset_name,
            target_label=params.get("target_label"),
            poisoning_ratio=params.get("poisoning_ratio", 0.2),
            epsilon=params.get("epsilon", 0.25),
            projection_type=params.get("projection_type", "l_2"),
            scaling_factor=params.get("scaling_factor", 1.0),
        )

    def poison_local_dataset(self, dataset, client_id: int, malicious_ids: List[int]):
        images, labels = _resolve_dataset_tensors(dataset)
        poisoned_images, poisoned_labels = self.attack.poison_batch(images, labels)
        return TensorDataset(poisoned_images, poisoned_labels)

    def poison_client_state(
        self,
        current_state: Dict[str, torch.Tensor],
        client_id: int,
        malicious_ids: List[int],
        benign_states,
        global_state_dict,
        num_clients: int,
    ) -> Dict[str, torch.Tensor]:
        # edge-case 的关键不只是投毒数据，还要把本地 update 放大成可主导聚合的后门更新。
        return _constrain_and_scale_local_state(
            current_state=current_state,
            global_state_dict=global_state_dict,
            epsilon=self.attack.epsilon,
            projection_type=self.attack.projection_type,
            scaling_factor=self.attack.scaling_factor,
        )

    def build_asr_eval_loader(self, clean_test_dataset, batch_size: int):
        return self.attack.get_poisoned_test_loader(batch_size=batch_size)


class SemanticBackdoorAdapter(BaseAttackAdapter):
    attack_name = "semantic_backdoor"
    attack_mode = "targeted"
    is_data_poisoning = True
    is_update_poisoning = True  # 语义后门在 FL 场景下同样需要 model replacement 才足够强

    def __init__(self, dataset_name: str, params: Dict[str, object]) -> None:
        default_source = "ardis" if dataset_name == "mnist" else "southwest"
        self.attack = SemanticBackdoorAttack(
            dataset_name=dataset_name,
            target_label=params.get("target_label", 0),
            poisoning_ratio=params.get("poisoning_ratio", 0.2),
            semantic_source=params.get("semantic_source", default_source),
            epsilon=params.get("epsilon", 0.25),
            projection_type=params.get("projection_type", "l_2"),
            scaling_factor=params.get("scaling_factor", 1.0),
        )

    def poison_local_dataset(self, dataset, client_id: int, malicious_ids: List[int]):
        return self.attack.poison_dataset(dataset, train=True)

    def poison_client_state(
        self,
        current_state: Dict[str, torch.Tensor],
        client_id: int,
        malicious_ids: List[int],
        benign_states,
        global_state_dict,
        num_clients: int,
    ) -> Dict[str, torch.Tensor]:
        # 仅注入语义样本往往不够，补上 constrain-and-scale 后更接近论文里的强语义后门。
        return _constrain_and_scale_local_state(
            current_state=current_state,
            global_state_dict=global_state_dict,
            epsilon=self.attack.epsilon,
            projection_type=self.attack.projection_type,
            scaling_factor=self.attack.scaling_factor,
        )

    def build_asr_eval_loader(self, clean_test_dataset, batch_size: int):
        return self.attack.get_poisoned_loader(
            clean_test_dataset,
            batch_size=batch_size,
            train=False,
            shuffle=False,
        )


class LabelFlippingTargetedAdapter(BaseAttackAdapter):
    attack_name = "label_flipping_targeted"
    attack_mode = "targeted"
    is_data_poisoning = True

    def __init__(self, params: Dict[str, object]) -> None:
        self.source_class = int(params.get("source_class", 1))
        self.target_class = int(params.get("target_class", 0))
        self.poison_ratio = float(params.get("poison_ratio", 1.0))

    def poison_local_dataset(self, dataset, client_id: int, malicious_ids: List[int]):
        images, labels = _resolve_dataset_tensors(dataset)
        poisoned_labels = label_flipping_targeted(
            labels.clone(),
            target_class=self.target_class,
            source_class=self.source_class,
            poison_ratio=self.poison_ratio,
            inplace=False,
        )
        return TensorDataset(images, poisoned_labels)

    def build_asr_eval_loader(self, clean_test_dataset, batch_size: int):
        images, labels = _resolve_dataset_tensors(clean_test_dataset)
        mask = labels == self.source_class
        if int(mask.sum().item()) == 0:
            return None
        target_labels = torch.full(
            (int(mask.sum().item()),),
            self.target_class,
            dtype=labels.dtype,
        )
        dataset = TensorDataset(images[mask].clone(), target_labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class LabelFlippingUntargetedAdapter(BaseAttackAdapter):
    attack_name = "label_flipping_untargeted"
    attack_mode = "untargeted"
    is_data_poisoning = True

    def __init__(self, num_classes: int, params: Dict[str, object]) -> None:
        self.num_classes = int(num_classes)
        self.poison_ratio = float(params.get("poison_ratio", 1.0))

    def poison_local_dataset(self, dataset, client_id: int, malicious_ids: List[int]):
        images, labels = _resolve_dataset_tensors(dataset)
        poisoned_labels = label_flipping_untargeted(
            labels.clone(),
            num_classes=self.num_classes,
            poison_ratio=self.poison_ratio,
            inplace=False,
        )
        return TensorDataset(images, poisoned_labels)


class SignFlippingAdapter(BaseAttackAdapter):
    attack_name = "sign_flipping"
    attack_mode = "untargeted"
    is_update_poisoning = True

    def __init__(self, params: Dict[str, object]) -> None:
        self.alpha = float(params.get("alpha", 1.0))

    def poison_client_state(self, current_state, client_id, malicious_ids, benign_states, global_state_dict, num_clients):
        return _poison_local_state_via_update(
            current_state=current_state,
            global_state_dict=global_state_dict,
            attack_fn=sign_flipping,
            alpha=self.alpha,
        )


class ScalingAttackAdapter(BaseAttackAdapter):
    attack_name = "scaling_attack"
    attack_mode = "untargeted"
    is_update_poisoning = True

    def __init__(self, params: Dict[str, object]) -> None:
        self.gamma = float(params.get("gamma", 10.0))

    def poison_client_state(self, current_state, client_id, malicious_ids, benign_states, global_state_dict, num_clients):
        return _poison_local_state_via_update(
            current_state=current_state,
            global_state_dict=global_state_dict,
            attack_fn=scaling_attack,
            gamma=self.gamma,
        )


class AdditiveNoiseAdapter(BaseAttackAdapter):
    attack_name = "additive_noise"
    attack_mode = "untargeted"
    is_update_poisoning = True

    def __init__(self, params: Dict[str, object]) -> None:
        self.mean = float(params.get("mean", 0.0))
        self.std = float(params.get("std", 0.1))

    def poison_client_state(self, current_state, client_id, malicious_ids, benign_states, global_state_dict, num_clients):
        return _poison_local_state_via_update(
            current_state=current_state,
            global_state_dict=global_state_dict,
            attack_fn=additive_noise_attack,
            mean=self.mean,
            std=self.std,
        )


class RandomGradientAdapter(BaseAttackAdapter):
    attack_name = "random_gradient"
    attack_mode = "untargeted"
    is_update_poisoning = True

    def __init__(self, params: Dict[str, object]) -> None:
        self.mean = float(params.get("mean", 0.0))
        self.std = float(params.get("std", 1.0))

    def poison_client_state(self, current_state, client_id, malicious_ids, benign_states, global_state_dict, num_clients):
        return _poison_local_state_via_update(
            current_state=current_state,
            global_state_dict=global_state_dict,
            attack_fn=random_gradient_attack,
            mean=self.mean,
            std=self.std,
        )


class ALIEAdapter(BaseAttackAdapter):
    attack_name = "a_lie"
    attack_mode = "untargeted"
    is_update_poisoning = True

    def __init__(self, params: Dict[str, object]) -> None:
        self.z_max = params.get("z_max")
        self.client_jitter_std = float(params.get("client_jitter_std", 0.0))
        self._stats_cache_token = None
        self._stats_cache = None

    def poison_client_state(self, current_state, client_id, malicious_ids, benign_states, global_state_dict, num_clients):
        if not benign_states:
            return current_state
        cache_token = (id(benign_states), id(global_state_dict), len(benign_states))
        if self._stats_cache_token != cache_token:
            # 同一轮内良性 update 分布固定，缓存均值/方差供多个恶意客户端复用。
            self._stats_cache = build_alie_update_stats(benign_states, global_state_dict)
            self._stats_cache_token = cache_token
        attack_update = alie_attack_update(
            benign_state_dicts=benign_states,
            global_state_dict=global_state_dict,
            num_clients=num_clients,
            num_adv=len(malicious_ids),
            z_max=self.z_max,
            client_id=client_id,
            client_jitter_std=self.client_jitter_std,
            update_stats=self._stats_cache,
        )
        return _rebuild_state_from_delta(current_state, global_state_dict, attack_update)


class FedImpAdapter(BaseAttackAdapter):
    attack_name = "fedimp"
    attack_mode = "untargeted"
    is_update_poisoning = True

    def __init__(self, params: Dict[str, object]) -> None:
        self.fedimp_factor = float(params.get("fedimp_factor", 2.0))
        self.top_k_ratio = float(params.get("top_k_ratio", 0.1))
        raw_compute_device = str(params.get("compute_device", "cpu")).strip().lower()
        self.compute_device = (
            "cuda"
            if raw_compute_device in {"auto", "cuda"} and torch.cuda.is_available()
            else "cpu"
        )
        dense_stats_max_mb = float(params.get("dense_stats_max_mb", 64.0))
        self.dense_stats_max_numel = max(1, int(dense_stats_max_mb * 1024 * 1024 / 4))
        self._stats_cache_token = None
        self._stats_cache = None

    def poison_client_state(self, current_state, client_id, malicious_ids, benign_states, global_state_dict, num_clients):
        simulated_updates = (
            LazyStateDeltaSequence(benign_states, global_state_dict)
            if benign_states and len(benign_states) > 0
            else None
        )
        simulated_update_stats = None
        if simulated_updates is not None:
            cache_token = (id(benign_states), id(global_state_dict), len(benign_states))
            if self._stats_cache_token != cache_token:
                # 同一轮内所有恶意客户端共享一组良性 update 统计量，
                # 避免每个恶意客户端都重复扫描完整 ResNet/VGG state；中小模型可把 dense 统计放到 GPU。
                self._stats_cache = build_fedimp_simulated_update_stats(
                    simulated_updates,
                    compute_device=self.compute_device,
                    max_dense_numel=self.dense_stats_max_numel,
                )
                self._stats_cache_token = cache_token
            simulated_update_stats = self._stats_cache
        return fedimp_attack(
            trained_state_dict=current_state,
            global_state_dict=global_state_dict,
            simulated_updates=simulated_updates,
            fedimp_factor=self.fedimp_factor,
            top_k_ratio=self.top_k_ratio,
            simulated_update_stats=simulated_update_stats,
        )


ATTACK_ADAPTERS = {
    "badnets": BadNetsAdapter,
    "dba": DBAAdapter,
    "edge_case": EdgeCaseAdapter,
    "semantic_backdoor": SemanticBackdoorAdapter,
    "label_flipping_targeted": LabelFlippingTargetedAdapter,
    "label_flipping_untargeted": LabelFlippingUntargetedAdapter,
    "sign_flipping": SignFlippingAdapter,
    "scaling_attack": ScalingAttackAdapter,
    "additive_noise": AdditiveNoiseAdapter,
    "random_gradient": RandomGradientAdapter,
    "a_lie": ALIEAdapter,
    "fedimp": FedImpAdapter,
}
