from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


_EXACT_THRESHOLD_MAX_NUMEL = 8_000_000
_DENSE_ATTACK_MAX_NUMEL = 64_000_000
_APPROX_THRESHOLD_BINS = 2048
_STACK_STATS_MAX_NUMEL = 16_000_000
_MIN_STD = 1e-5


def _resolve_compute_device(compute_device: str | torch.device) -> torch.device:
    normalized_device = str(compute_device).strip().lower()
    if normalized_device in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_stream_stats_device(
    compute_device: str | torch.device,
    total_numel: int,
    max_dense_numel: int,
) -> torch.device:
    target_device = _resolve_compute_device(compute_device)
    if target_device.type != "cuda":
        return torch.device("cpu")

    # fallback 路径只常驻 sum/sumsq 两份全模型统计量；
    # 用同一个预算控制 GPU 峰值，避免为省时间把显存打满。
    if int(total_numel) * 2 <= int(max_dense_numel):
        return target_device
    return torch.device("cpu")


def _topk_threshold(values: Tensor, k_num: int) -> Tensor:
    if int(k_num) <= 0:
        return torch.tensor(float("inf"), device=values.device, dtype=values.dtype)
    if int(k_num) >= int(values.numel()):
        return values.min()
    # 只需要第 k 大阈值，不需要 top-k 内部有序；这能减少大向量上的排序开销。
    return torch.topk(values, int(k_num), sorted=False).values.min()


@dataclass
class FedImpUpdateStats:
    """良性客户端 update 的逐层流式统计量。"""

    mean: Dict[str, Tensor]
    std: Dict[str, Tensor]
    count: Dict[str, int]
    flat_keys: Optional[Tuple[str, ...]] = None
    flat_shapes: Optional[Dict[str, torch.Size]] = None
    flat_sizes: Optional[Dict[str, int]] = None
    mean_flat: Optional[Tensor] = None
    std_flat: Optional[Tensor] = None


def _validate_current_update_dict(current_update_dict: Dict[str, Tensor]) -> None:
    if not isinstance(current_update_dict, dict):
        raise TypeError("current_update_dict 必须是 Dict[str, Tensor]。")

    for key, update in current_update_dict.items():
        if not torch.is_tensor(update):
            raise TypeError(f"current_update_dict['{key}'] 不是 Tensor。")
        if update.shape is None:
            raise ValueError(f"current_update_dict['{key}'] 形状非法。")


def _build_current_update_dict(
    trained_state_dict: Dict[str, Tensor],
    global_state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    current_update_dict: Dict[str, Tensor] = {}
    for key, param in trained_state_dict.items():
        if not torch.is_tensor(param):
            raise TypeError(f"{key} 对应的 trained 参数不是 Tensor。")
        if key not in global_state_dict:
            raise KeyError(f"global_state_dict 缺少 key: {key}")

        global_param = global_state_dict[key]
        if not torch.is_tensor(global_param):
            raise TypeError(f"{key} 对应的 global 参数不是 Tensor。")
        if param.shape != global_param.shape:
            raise ValueError(f"{key} 的 local/global 形状不一致。")
        if not param.dtype.is_floating_point:
            continue

        current_update_dict[key] = (
            param.detach().to(device="cpu", dtype=torch.float32)
            - global_param.detach().to(device="cpu", dtype=torch.float32)
        )

    return current_update_dict


def _resolve_fisher_tensor(
    key: str,
    current_update: Tensor,
    fisher_info: Optional[Dict[str, Tensor]],
) -> Tensor:
    if fisher_info is not None and key in fisher_info:
        fisher_tensor = fisher_info[key]
        if not torch.is_tensor(fisher_tensor):
            raise TypeError(f"fisher_info['{key}'] 不是 Tensor。")
        if fisher_tensor.shape != current_update.shape:
            raise ValueError(f"fisher_info['{key}'] 形状与当前更新不一致。")
        if fisher_tensor.dtype.is_floating_point:
            return fisher_tensor.detach().to(device="cpu", dtype=torch.float32)

    return current_update.detach().to(device="cpu", dtype=torch.float32).pow(2)


def _resolve_fisher_tensor_on_device(
    key: str,
    current_update: Tensor,
    fisher_info: Optional[Dict[str, Tensor]],
    target_device: torch.device,
) -> Tensor:
    if fisher_info is not None and key in fisher_info:
        fisher_tensor = fisher_info[key]
        if not torch.is_tensor(fisher_tensor):
            raise TypeError(f"fisher_info['{key}'] 不是 Tensor。")
        if fisher_tensor.shape != current_update.shape:
            raise ValueError(f"fisher_info['{key}'] 形状与当前更新不一致。")
        if fisher_tensor.dtype.is_floating_point:
            return fisher_tensor.detach().to(
                device=target_device,
                dtype=torch.float32,
                non_blocking=target_device.type == "cuda",
            )

    return current_update.detach().to(
        device=target_device,
        dtype=torch.float32,
        non_blocking=target_device.type == "cuda",
    ).pow(2)


def _resolve_topk_threshold(
    current_update_dict: Dict[str, Tensor],
    top_k_ratio: float,
    fisher_info: Optional[Dict[str, Tensor]],
) -> float:
    total_numel = sum(update.numel() for update in current_update_dict.values())
    if total_numel == 0:
        return float("inf")

    k_num = int(total_numel * top_k_ratio)
    if top_k_ratio > 0.0 and k_num == 0:
        k_num = 1
    if k_num <= 0:
        return float("inf")

    if total_numel <= _EXACT_THRESHOLD_MAX_NUMEL:
        fisher_values = [
            _resolve_fisher_tensor(key, update, fisher_info).reshape(-1)
            for key, update in current_update_dict.items()
        ]
        all_fisher = torch.cat(fisher_values, dim=0)
        return float(_topk_threshold(all_fisher, k_num).item())

    max_log = 0.0
    for key, update in current_update_dict.items():
        fisher = _resolve_fisher_tensor(key, update, fisher_info)
        if fisher.numel() == 0:
            continue
        layer_max = float(torch.log1p(fisher.max()).item())
        if layer_max > max_log:
            max_log = layer_max

    if max_log <= 0.0:
        return 0.0

    hist = torch.zeros(_APPROX_THRESHOLD_BINS, dtype=torch.int64)
    denom = max_log if max_log > 0 else 1.0

    for key, update in current_update_dict.items():
        fisher = _resolve_fisher_tensor(key, update, fisher_info).reshape(-1)
        if fisher.numel() == 0:
            continue
        fisher_log = torch.log1p(fisher)
        bin_indices = torch.clamp(
            (fisher_log / denom * (_APPROX_THRESHOLD_BINS - 1)).long(),
            min=0,
            max=_APPROX_THRESHOLD_BINS - 1,
        )
        hist += torch.bincount(bin_indices, minlength=_APPROX_THRESHOLD_BINS)

    remaining = k_num
    for index in range(_APPROX_THRESHOLD_BINS - 1, -1, -1):
        remaining -= int(hist[index].item())
        if remaining <= 0:
            threshold_log = denom * float(index) / float(_APPROX_THRESHOLD_BINS - 1)
            return float(torch.expm1(torch.tensor(threshold_log)).item())

    return 0.0


def _resolve_topk_threshold_with_fisher_cache(
    current_update_dict: Dict[str, Tensor],
    top_k_ratio: float,
    fisher_info: Optional[Dict[str, Tensor]],
) -> Tuple[float, Dict[str, Tensor]]:
    total_numel = sum(update.numel() for update in current_update_dict.values())
    if total_numel == 0:
        return float("inf"), {}

    k_num = int(total_numel * top_k_ratio)
    if top_k_ratio > 0.0 and k_num == 0:
        k_num = 1
    if k_num <= 0:
        return float("inf"), {}

    if total_numel > _EXACT_THRESHOLD_MAX_NUMEL:
        # 大模型继续走直方图近似，避免为了缓存 Fisher 再额外常驻一份全模型张量。
        return _resolve_topk_threshold(
            current_update_dict=current_update_dict,
            top_k_ratio=top_k_ratio,
            fisher_info=fisher_info,
        ), {}

    fisher_cache = {
        key: _resolve_fisher_tensor(key, update, fisher_info)
        for key, update in current_update_dict.items()
    }
    all_fisher = torch.cat([fisher.reshape(-1) for fisher in fisher_cache.values()], dim=0)
    threshold = float(_topk_threshold(all_fisher, k_num).item())
    return threshold, fisher_cache


def _stream_simulated_update_stats(
    simulated_updates: Sequence[Dict[str, Tensor]],
    key: str,
    fallback_update: Tensor,
) -> Tuple[Tensor, Tensor]:
    mean = None
    m2 = None
    count = 0

    for i, sim_update in enumerate(simulated_updates):
        if not isinstance(sim_update, dict):
            raise TypeError(f"simulated_updates[{i}] 不是 Dict[str, Tensor]。")
        if key not in sim_update:
            continue

        sim_tensor = sim_update[key]
        if not torch.is_tensor(sim_tensor):
            raise TypeError(f"simulated_updates[{i}]['{key}'] 不是 Tensor。")
        if not sim_tensor.dtype.is_floating_point:
            continue
        if sim_tensor.shape != fallback_update.shape:
            raise ValueError(f"simulated_updates[{i}]['{key}'] 形状与当前更新不一致。")

        value = sim_tensor.detach().to(device="cpu", dtype=torch.float32)
        count += 1

        if mean is None:
            mean = torch.zeros_like(value)
            m2 = torch.zeros_like(value)

        delta = value - mean
        mean = mean + delta / float(count)
        delta2 = value - mean
        m2 = m2 + delta * delta2

    if count > 1:
        variance = m2 / float(count - 1)
        variance.clamp_(min=0.0)
        return mean, torch.sqrt(variance)
    if count == 1:
        return mean, torch.zeros_like(mean) + _MIN_STD
    return fallback_update, torch.zeros_like(fallback_update) + _MIN_STD


def _copy_update_to_dense_row(
    matrix: Tensor,
    row_index: int,
    sim_update: Dict[str, Tensor],
    keys: Sequence[str],
    shapes: Dict[str, torch.Size],
    sizes: Dict[str, int],
) -> None:
    cursor = 0
    for key in keys:
        if key not in sim_update:
            raise KeyError(f"simulated_update 缺少 key: {key}")
        sim_tensor = sim_update[key]
        if not torch.is_tensor(sim_tensor):
            raise TypeError(f"simulated_update['{key}'] 不是 Tensor。")
        if not sim_tensor.dtype.is_floating_point:
            raise TypeError(f"simulated_update['{key}'] 不是浮点 Tensor。")
        if sim_tensor.shape != shapes[key]:
            raise ValueError(f"simulated_update['{key}'] 形状与已有统计不一致。")

        size = sizes[key]
        target_device = matrix.device
        matrix[row_index, cursor : cursor + size].copy_(
            sim_tensor.detach().to(
                device=target_device,
                dtype=torch.float32,
                non_blocking=target_device.type == "cuda",
            ).reshape(-1)
        )
        cursor += size


def _try_build_dense_simulated_update_stats(
    simulated_updates: Sequence[Dict[str, Tensor]],
    compute_device: str | torch.device = "cpu",
    max_dense_numel: int = _STACK_STATS_MAX_NUMEL,
) -> Optional[FedImpUpdateStats]:
    update_count = len(simulated_updates)
    if update_count <= 0:
        return None

    target_device = _resolve_compute_device(compute_device)
    dense_builder = getattr(simulated_updates, "build_dense_update_matrix", None)
    if callable(dense_builder):
        dense_payload = dense_builder(max_numel=max_dense_numel, device=target_device)
        if dense_payload is not None:
            matrix, keys, shapes, sizes = dense_payload
            return _build_stats_from_dense_matrix(
                matrix=matrix,
                keys=keys,
                shapes=shapes,
                sizes=sizes,
                update_count=update_count,
            )

    first_update = simulated_updates[0]
    if not isinstance(first_update, dict):
        raise TypeError("simulated_updates[0] 不是 Dict[str, Tensor]。")

    keys = [
        key
        for key, tensor in first_update.items()
        if torch.is_tensor(tensor) and tensor.dtype.is_floating_point
    ]
    if not keys:
        return FedImpUpdateStats(mean={}, std={}, count={})

    shapes = {key: first_update[key].shape for key in keys}
    sizes = {key: int(first_update[key].numel()) for key in keys}
    total_numel = int(sum(sizes.values()))
    if total_numel * update_count > int(max_dense_numel):
        return None

    # 中小模型直接构造“客户端 x 参数”的 dense update 矩阵；
    # 统计只扫一次矩阵，避免按层启动大量小 Tensor 运算；配置为 CUDA 时整块统计留在 GPU。
    matrix = torch.empty((update_count, total_numel), dtype=torch.float32, device=target_device)
    _copy_update_to_dense_row(
        matrix=matrix,
        row_index=0,
        sim_update=first_update,
        keys=keys,
        shapes=shapes,
        sizes=sizes,
    )
    for update_index in range(1, update_count):
        sim_update = simulated_updates[update_index]
        if not isinstance(sim_update, dict):
            raise TypeError(f"simulated_updates[{update_index}] 不是 Dict[str, Tensor]。")
        _copy_update_to_dense_row(
            matrix=matrix,
            row_index=update_index,
            sim_update=sim_update,
            keys=keys,
            shapes=shapes,
            sizes=sizes,
        )

    return _build_stats_from_dense_matrix(
        matrix=matrix,
        keys=keys,
        shapes=shapes,
        sizes=sizes,
        update_count=update_count,
    )


def _build_stats_from_dense_matrix(
    matrix: Tensor,
    keys: Sequence[str],
    shapes: Dict[str, torch.Size],
    sizes: Dict[str, int],
    update_count: int,
) -> FedImpUpdateStats:
    if not keys:
        return FedImpUpdateStats(mean={}, std={}, count={})

    if update_count > 1:
        variance_flat, mean_flat = torch.var_mean(matrix, dim=0, unbiased=True)
        std_flat = torch.sqrt(variance_flat.clamp_(min=0.0))
    else:
        mean_flat = matrix[0]
        std_flat = torch.zeros_like(mean_flat) + _MIN_STD

    mean_flat_storage = mean_flat.clone()
    std_flat_storage = std_flat.clone()
    means: Dict[str, Tensor] = {}
    std_values: Dict[str, Tensor] = {}
    counts: Dict[str, int] = {}
    cursor = 0
    for key in keys:
        size = sizes[key]
        # mean/std 字典直接引用 flat 缓存的视图；
        # dense 路径后续攻击会优先使用 flat 统计，避免再额外常驻一份逐层 clone。
        means[key] = mean_flat_storage[cursor : cursor + size].reshape(shapes[key])
        std_values[key] = std_flat_storage[cursor : cursor + size].reshape(shapes[key])
        counts[key] = update_count
        cursor += size
    return FedImpUpdateStats(
        mean=means,
        std=std_values,
        count=counts,
        flat_keys=tuple(keys),
        flat_shapes=dict(shapes),
        flat_sizes=dict(sizes),
        mean_flat=mean_flat_storage,
        std_flat=std_flat_storage,
    )


def _build_stats_from_sums(
    sums: Dict[str, Tensor],
    sumsq_values: Dict[str, Tensor],
    counts: Dict[str, int],
) -> FedImpUpdateStats:
    means: Dict[str, Tensor] = {}
    std_values: Dict[str, Tensor] = {}
    for key, sum_tensor in sums.items():
        count = counts[key]
        if count <= 0:
            continue

        mean = sum_tensor.div_(float(count))
        means[key] = mean
        if count > 1:
            variance = sumsq_values[key]
            variance.addcmul_(mean, mean, value=-float(count))
            variance.div_(float(count - 1)).clamp_(min=0.0)
            std_values[key] = variance.sqrt_()
        else:
            std_values[key] = torch.zeros_like(mean) + _MIN_STD
    return FedImpUpdateStats(mean=means, std=std_values, count=counts)


def build_fedimp_simulated_update_stats(
    simulated_updates: Sequence[Dict[str, Tensor]],
    compute_device: str | torch.device = "cpu",
    max_dense_numel: int = _STACK_STATS_MAX_NUMEL,
) -> FedImpUpdateStats:
    """
    一次性流式统计所有良性客户端 update 的均值和标准差。

    FedImp 原实现会在“每个恶意客户端 × 每一层”里重复遍历良性客户端，
    ResNet34 这类大模型会因此反复构造完整 delta。这里把统计量提升为 round 级缓存。
    """
    if simulated_updates is None or len(simulated_updates) == 0:
        raise ValueError("simulated_updates 至少包含一个良性客户端 update。")

    dense_stats = _try_build_dense_simulated_update_stats(
        simulated_updates,
        compute_device=compute_device,
        max_dense_numel=max_dense_numel,
    )
    if dense_stats is not None:
        return dense_stats

    metadata_builder = getattr(simulated_updates, "get_float_metadata", None)
    sum_sumsq_builder = getattr(simulated_updates, "build_update_sum_sumsq", None)
    if callable(metadata_builder) and callable(sum_sumsq_builder):
        keys, shapes, sizes = metadata_builder()
        total_numel = int(sum(int(size) for size in sizes.values()))
        target_device = _resolve_stream_stats_device(
            compute_device=compute_device,
            total_numel=total_numel,
            max_dense_numel=max_dense_numel,
        )
        sums, sumsq_values, counts, _ = sum_sumsq_builder(
            device=target_device,
            keys=keys,
            shapes=shapes,
        )
        # LazyStateDeltaSequence 可直接流式产出一阶/二阶和；
        # 这条快路径省掉每个 benign 客户端完整 delta dict 的构造成本。
        return _build_stats_from_sums(
            sums=sums,
            sumsq_values=sumsq_values,
            counts=counts,
        )

    first_update = simulated_updates[0]
    if not isinstance(first_update, dict):
        raise TypeError("simulated_updates[0] 不是 Dict[str, Tensor]。")
    first_float_tensors = [
        tensor
        for tensor in first_update.values()
        if torch.is_tensor(tensor) and tensor.dtype.is_floating_point
    ]
    total_numel = int(sum(int(tensor.numel()) for tensor in first_float_tensors))
    target_device = _resolve_stream_stats_device(
        compute_device=compute_device,
        total_numel=total_numel,
        max_dense_numel=max_dense_numel,
    )
    del first_float_tensors

    sums: Dict[str, Tensor] = {}
    sumsq_values: Dict[str, Tensor] = {}
    counts: Dict[str, int] = {}
    shapes: Dict[str, torch.Size] = {}

    for update_index in range(len(simulated_updates)):
        sim_update = first_update if update_index == 0 else simulated_updates[update_index]
        if not isinstance(sim_update, dict):
            raise TypeError(f"simulated_updates[{update_index}] 不是 Dict[str, Tensor]。")

        for key, sim_tensor in sim_update.items():
            if not torch.is_tensor(sim_tensor):
                raise TypeError(f"simulated_updates[{update_index}]['{key}'] 不是 Tensor。")
            if not sim_tensor.dtype.is_floating_point:
                continue

            if key not in shapes:
                shapes[key] = sim_tensor.shape
                counts[key] = 0
                sums[key] = torch.zeros_like(sim_tensor, device=target_device, dtype=torch.float32)
                sumsq_values[key] = torch.zeros_like(sim_tensor, device=target_device, dtype=torch.float32)
            elif shapes[key] != sim_tensor.shape:
                raise ValueError(f"simulated_updates[{update_index}]['{key}'] 形状与已有统计不一致。")

            value = sim_tensor.detach().to(
                device=target_device,
                dtype=torch.float32,
                non_blocking=target_device.type == "cuda",
            )
            # 大模型 fallback 只保留 sum/sumsq 两份统计量；
            # 相比把所有 benign update 攒起来再 stack，峰值内存与客户端数解耦。
            counts[key] += 1
            sums[key].add_(value)
            sumsq_values[key].addcmul_(value, value)

    return _build_stats_from_sums(
        sums=sums,
        sumsq_values=sumsq_values,
        counts=counts,
    )


def _proxy_scale_stats(proxy_scales: Sequence[float]) -> Tuple[float, float]:
    if proxy_scales is None or len(proxy_scales) == 0:
        raise ValueError("proxy_scales 至少包含一个缩放系数。")

    scales = torch.tensor(list(proxy_scales), dtype=torch.float32)
    mean_scale = float(scales.mean().item())
    std_scale = float(scales.std(unbiased=True).item()) if scales.numel() > 1 else 0.0
    return mean_scale, std_scale


def _try_fedimp_attack_update_dense(
    current_update_dict: Dict[str, Tensor],
    simulated_update_stats: Optional[FedImpUpdateStats],
    fedimp_factor: float,
    top_k_ratio: float,
    fisher_info: Optional[Dict[str, Tensor]],
) -> Optional[Dict[str, Tensor]]:
    if simulated_update_stats is None:
        return None

    keys = list(current_update_dict.keys())
    total_numel = int(sum(int(current_update_dict[key].numel()) for key in keys))
    if total_numel == 0:
        return {}

    for key in keys:
        if key not in simulated_update_stats.mean or key not in simulated_update_stats.std:
            return None
        if simulated_update_stats.mean[key].shape != current_update_dict[key].shape:
            return None
        if simulated_update_stats.std[key].shape != current_update_dict[key].shape:
            return None
        if fisher_info is not None and key in fisher_info and fisher_info[key].shape != current_update_dict[key].shape:
            return None

    first_stat_tensor = next(iter(simulated_update_stats.mean.values()), None)
    target_device = simulated_update_stats.mean_flat.device if simulated_update_stats.mean_flat is not None else (
        first_stat_tensor.device if first_stat_tensor is not None else torch.device("cpu")
    )
    max_dense_attack_numel = (
        _DENSE_ATTACK_MAX_NUMEL
        if target_device.type == "cuda"
        else _EXACT_THRESHOLD_MAX_NUMEL
    )
    if total_numel > max_dense_attack_numel:
        return None
    importance_flat = torch.empty(total_numel, dtype=torch.float32, device=target_device)
    can_reuse_flat_stats = (
        simulated_update_stats.flat_keys == tuple(keys)
        and simulated_update_stats.mean_flat is not None
        and simulated_update_stats.std_flat is not None
        and int(simulated_update_stats.mean_flat.numel()) == total_numel
        and int(simulated_update_stats.std_flat.numel()) == total_numel
        and simulated_update_stats.mean_flat.device == target_device
        and simulated_update_stats.std_flat.device == target_device
    )
    if can_reuse_flat_stats:
        # dense 统计阶段已经缓存了 flat mu/std，同一轮多个恶意客户端无需再按层拼接。
        mu_flat = simulated_update_stats.mean_flat.detach().to(device=target_device, dtype=torch.float32)
        sigma_flat = simulated_update_stats.std_flat.detach().to(device=target_device, dtype=torch.float32)
    else:
        mu_flat = torch.empty(total_numel, dtype=torch.float32, device=target_device)
        sigma_flat = torch.empty(total_numel, dtype=torch.float32, device=target_device)

    cursor = 0
    shapes: Dict[str, torch.Size] = {}
    sizes: Dict[str, int] = {}
    for key in keys:
        update_tensor = current_update_dict[key].detach().to(
            device=target_device,
            dtype=torch.float32,
            non_blocking=target_device.type == "cuda",
        ).reshape(-1)
        size = int(update_tensor.numel())
        shapes[key] = current_update_dict[key].shape
        sizes[key] = size
        if not can_reuse_flat_stats:
            mu_flat[cursor : cursor + size].copy_(
                simulated_update_stats.mean[key].detach().to(
                    device=target_device,
                    dtype=torch.float32,
                    non_blocking=target_device.type == "cuda",
                ).reshape(-1)
            )
            sigma_flat[cursor : cursor + size].copy_(
                simulated_update_stats.std[key].detach().to(
                    device=target_device,
                    dtype=torch.float32,
                    non_blocking=target_device.type == "cuda",
                ).reshape(-1)
            )
        if fisher_info is not None:
            importance_flat[cursor : cursor + size].copy_(
                _resolve_fisher_tensor_on_device(
                    key,
                    current_update_dict[key],
                    fisher_info,
                    target_device,
                ).reshape(-1)
            )
        else:
            # Fisher 缺省时使用 update^2，排序上等价于 abs(update)；
            # 避免额外 materialize 一份平方后的全模型向量。
            torch.abs(update_tensor, out=importance_flat[cursor : cursor + size])
        cursor += size

    k_num = int(total_numel * top_k_ratio)
    if top_k_ratio > 0.0 and k_num == 0:
        k_num = 1

    if k_num <= 0:
        poisoned_flat = mu_flat
    else:
        # 中小模型用一条 flat mask 生成攻击更新，避免对每层分别创建 mask 和小 Tensor 算子。
        threshold = _topk_threshold(importance_flat, k_num)
        mask_flat = importance_flat >= threshold
        # 复用临时拼接出来的 mu_flat；只有 flat 统计被缓存复用时才 clone，避免污染后续恶意客户端。
        poisoned_flat = mu_flat.clone() if can_reuse_flat_stats else mu_flat
        poisoned_flat[mask_flat] -= float(fedimp_factor) * sigma_flat[mask_flat]

    poisoned_update_dict: Dict[str, Tensor] = {}
    cursor = 0
    for key in keys:
        size = sizes[key]
        poisoned_update_dict[key] = poisoned_flat[cursor : cursor + size].reshape(shapes[key]).clone()
        cursor += size
    return poisoned_update_dict


def _try_fedimp_attack_state_dense(
    trained_state_dict: Dict[str, Tensor],
    global_state_dict: Dict[str, Tensor],
    simulated_update_stats: Optional[FedImpUpdateStats],
    fedimp_factor: float,
    top_k_ratio: float,
    fisher_info: Optional[Dict[str, Tensor]],
) -> Optional[Dict[str, Tensor]]:
    if simulated_update_stats is None or fisher_info is not None:
        return None

    keys = [
        key
        for key, value in trained_state_dict.items()
        if torch.is_tensor(value) and value.dtype.is_floating_point
    ]
    total_numel = int(sum(int(trained_state_dict[key].numel()) for key in keys))
    if total_numel == 0:
        return None

    for key in keys:
        if key not in global_state_dict:
            raise KeyError(f"global_state_dict 缺少 key: {key}")
        if key not in simulated_update_stats.mean or key not in simulated_update_stats.std:
            return None
        if simulated_update_stats.mean[key].shape != trained_state_dict[key].shape:
            return None
        if simulated_update_stats.std[key].shape != trained_state_dict[key].shape:
            return None
        if not torch.is_tensor(global_state_dict[key]):
            raise TypeError(f"global_state_dict['{key}'] 不是 Tensor。")
        if global_state_dict[key].shape != trained_state_dict[key].shape:
            raise ValueError(f"{key} 的 local/global 形状不一致。")

    first_stat_tensor = next(iter(simulated_update_stats.mean.values()), None)
    target_device = simulated_update_stats.mean_flat.device if simulated_update_stats.mean_flat is not None else (
        first_stat_tensor.device if first_stat_tensor is not None else torch.device("cpu")
    )
    max_dense_attack_numel = (
        _DENSE_ATTACK_MAX_NUMEL
        if target_device.type == "cuda"
        else _EXACT_THRESHOLD_MAX_NUMEL
    )
    if total_numel > max_dense_attack_numel:
        return None

    can_reuse_flat_stats = (
        simulated_update_stats.flat_keys == tuple(keys)
        and simulated_update_stats.mean_flat is not None
        and simulated_update_stats.std_flat is not None
        and int(simulated_update_stats.mean_flat.numel()) == total_numel
        and int(simulated_update_stats.std_flat.numel()) == total_numel
        and simulated_update_stats.mean_flat.device == target_device
        and simulated_update_stats.std_flat.device == target_device
    )
    if can_reuse_flat_stats:
        mu_flat = simulated_update_stats.mean_flat.detach().to(device=target_device, dtype=torch.float32)
        sigma_flat = simulated_update_stats.std_flat.detach().to(device=target_device, dtype=torch.float32)
    else:
        mu_flat = torch.empty(total_numel, dtype=torch.float32, device=target_device)
        sigma_flat = torch.empty(total_numel, dtype=torch.float32, device=target_device)

    importance_flat = torch.empty(total_numel, dtype=torch.float32, device=target_device)
    shapes: Dict[str, torch.Size] = {}
    sizes: Dict[str, int] = {}
    cursor = 0
    for key in keys:
        local_value = trained_state_dict[key]
        global_value = global_state_dict[key]
        size = int(local_value.numel())
        shapes[key] = local_value.shape
        sizes[key] = size
        delta_view = importance_flat[cursor : cursor + size].reshape(local_value.shape)
        torch.sub(
            local_value.detach().to(
                device=target_device,
                dtype=torch.float32,
                non_blocking=target_device.type == "cuda",
            ),
            global_value.detach().to(
                device=target_device,
                dtype=torch.float32,
                non_blocking=target_device.type == "cuda",
            ),
            out=delta_view,
        )
        # 有缓存统计量时直接在 state 级别走 flat top-k；
        # Fisher 缺省用 |update| 排序，避免先构造 CPU update dict 再重复扫描。
        torch.abs(delta_view, out=delta_view)
        if not can_reuse_flat_stats:
            mu_flat[cursor : cursor + size].copy_(
                simulated_update_stats.mean[key].detach().to(
                    device=target_device,
                    dtype=torch.float32,
                    non_blocking=target_device.type == "cuda",
                ).reshape(-1)
            )
            sigma_flat[cursor : cursor + size].copy_(
                simulated_update_stats.std[key].detach().to(
                    device=target_device,
                    dtype=torch.float32,
                    non_blocking=target_device.type == "cuda",
                ).reshape(-1)
            )
        cursor += size

    k_num = int(total_numel * top_k_ratio)
    if top_k_ratio > 0.0 and k_num == 0:
        k_num = 1
    if k_num <= 0:
        poisoned_flat = mu_flat.clone() if can_reuse_flat_stats else mu_flat
    else:
        threshold = _topk_threshold(importance_flat, k_num)
        mask_flat = importance_flat >= threshold
        poisoned_flat = mu_flat.clone() if can_reuse_flat_stats else mu_flat
        poisoned_flat[mask_flat] -= float(fedimp_factor) * sigma_flat[mask_flat]

    poisoned_state_dict: Dict[str, Tensor] = {}
    cursor = 0
    float_key_set = set(keys)
    for key, value in trained_state_dict.items():
        if key not in float_key_set:
            poisoned_state_dict[key] = value.clone() if torch.is_tensor(value) else value
            continue

        size = sizes[key]
        global_value = global_state_dict[key]
        poisoned_update = poisoned_flat[cursor : cursor + size].reshape(shapes[key])
        poisoned_state_dict[key] = global_value + poisoned_update.to(
            device=global_value.device,
            dtype=global_value.dtype,
        )
        cursor += size
    return poisoned_state_dict


def fedimp_attack_update(
    current_update_dict: Dict[str, Tensor],
    simulated_updates: Optional[Sequence[Dict[str, Tensor]]] = None,
    fedimp_factor: float = 2.0,
    top_k_ratio: float = 0.1,
    fisher_info: Optional[Dict[str, Tensor]] = None,
    simulated_update_stats: Optional[FedImpUpdateStats] = None,
) -> Dict[str, Tensor]:
    """
    FedIMP 参数重要性模型投毒攻击 (Eq. 12):
        Δw_mal = μ - δ * M * σ

    参数:
        current_update_dict: 当前恶意客户端的上传更新
        simulated_updates: 本地模拟得到的良性更新列表(每个元素是 Dict[str, Tensor] 的更新)
        fedimp_factor: 攻击增强系数 δ
        top_k_ratio: Fisher Top-k 比例，用于构造全局掩码 M。
            小模型上精确求阈值；大模型上改为分桶近似，避免为阈值构造超大拼接向量。
        fisher_info: 预计算 Fisher 信息；若不传则用 update^2 近似
        simulated_update_stats: 已缓存的良性 update 统计量；传入后不再重复扫描 simulated_updates

    返回:
        恶意客户端应上传的模型更新字典
    """
    _validate_current_update_dict(current_update_dict)
    if simulated_updates is not None and not isinstance(simulated_updates, Sequence):
        raise TypeError("simulated_updates 必须是 Sequence[Dict[str, Tensor]] 或 None。")
    if fisher_info is not None and not isinstance(fisher_info, dict):
        raise TypeError("fisher_info 必须是 Dict[str, Tensor] 或 None。")
    if fedimp_factor < 0:
        raise ValueError("fedimp_factor 不能为负数。")
    if not (0.0 <= top_k_ratio <= 1.0):
        raise ValueError("top_k_ratio 必须在 [0.0, 1.0] 范围内。")

    # 没有可攻击参数时，直接返回空更新
    if len(current_update_dict) == 0:
        return {}

    dense_update = _try_fedimp_attack_update_dense(
        current_update_dict=current_update_dict,
        simulated_update_stats=simulated_update_stats,
        fedimp_factor=fedimp_factor,
        top_k_ratio=top_k_ratio,
        fisher_info=fisher_info,
    )
    if dense_update is not None:
        return dense_update

    threshold, fisher_cache = _resolve_topk_threshold_with_fisher_cache(
        current_update_dict=current_update_dict,
        top_k_ratio=top_k_ratio,
        fisher_info=fisher_info,
    )

    poisoned_update_dict: Dict[str, Tensor] = {}
    for key, current_update in current_update_dict.items():
        if simulated_update_stats is not None and key in simulated_update_stats.mean:
            mu = simulated_update_stats.mean[key]
            sigma = simulated_update_stats.std[key]
            if mu.shape != current_update.shape or sigma.shape != current_update.shape:
                raise ValueError(f"simulated_update_stats['{key}'] 形状与当前更新不一致。")
        elif simulated_updates and len(simulated_updates) > 0:
            update_cpu = current_update.detach().to(device="cpu", dtype=torch.float32)
            mu, sigma = _stream_simulated_update_stats(
                simulated_updates=simulated_updates,
                key=key,
                fallback_update=update_cpu,
            )
        else:
            mu = current_update.detach().to(device="cpu", dtype=torch.float32)
            sigma = torch.zeros_like(mu) + _MIN_STD

        target_device = mu.device
        fisher = fisher_cache.get(key)
        if fisher is None:
            fisher = _resolve_fisher_tensor_on_device(
                key=key,
                current_update=current_update,
                fisher_info=fisher_info,
                target_device=target_device,
            )
        elif fisher.device != target_device:
            fisher = fisher.to(device=target_device, non_blocking=target_device.type == "cuda")

        # 保持 mask 为 bool 并只更新 top-k 位置，避免为大模型再创建一份 float mask。
        mask = fisher >= threshold
        poisoned_update = mu.clone()
        poisoned_update[mask] -= float(fedimp_factor) * sigma[mask]
        poisoned_update_dict[key] = poisoned_update

    return poisoned_update_dict


def fedimp_attack_update_from_proxy(
    current_update_dict: Dict[str, Tensor],
    proxy_scales: Sequence[float],
    fedimp_factor: float = 2.0,
    top_k_ratio: float = 0.1,
    fisher_info: Optional[Dict[str, Tensor]] = None,
) -> Dict[str, Tensor]:
    """
    代理良性更新采用统一缩放时，直接从缩放统计构造 FedIMP 上传更新。

    这样无需真正物化 N 份完整模拟更新，能显著降低大模型攻击时的峰值内存。
    """
    _validate_current_update_dict(current_update_dict)
    if fedimp_factor < 0:
        raise ValueError("fedimp_factor 不能为负数。")
    if not (0.0 <= top_k_ratio <= 1.0):
        raise ValueError("top_k_ratio 必须在 [0.0, 1.0] 范围内。")

    mean_scale, std_scale = _proxy_scale_stats(proxy_scales)
    threshold = _resolve_topk_threshold(
        current_update_dict=current_update_dict,
        top_k_ratio=top_k_ratio,
        fisher_info=fisher_info,
    )

    poisoned_update_dict: Dict[str, Tensor] = {}
    for key, current_update in current_update_dict.items():
        update_cpu = current_update.detach().to(device="cpu", dtype=torch.float32)
        mu = update_cpu * mean_scale
        sigma = update_cpu.abs() * std_scale
        fisher = _resolve_fisher_tensor(key, update_cpu, fisher_info)
        mask = (fisher >= threshold).to(update_cpu.dtype)
        poisoned_update_dict[key] = mu - fedimp_factor * mask * sigma

    return poisoned_update_dict


def fedimp_attack(
    trained_state_dict: Dict[str, Tensor],
    global_state_dict: Dict[str, Tensor],
    simulated_updates: Optional[Sequence[Dict[str, Tensor]]] = None,
    fedimp_factor: float = 2.0,
    top_k_ratio: float = 0.1,
    fisher_info: Optional[Dict[str, Tensor]] = None,
    simulated_update_stats: Optional[FedImpUpdateStats] = None,
) -> Dict[str, Tensor]:
    dense_state = _try_fedimp_attack_state_dense(
        trained_state_dict=trained_state_dict,
        global_state_dict=global_state_dict,
        simulated_update_stats=simulated_update_stats,
        fedimp_factor=fedimp_factor,
        top_k_ratio=top_k_ratio,
        fisher_info=fisher_info,
    )
    if dense_state is not None:
        return dense_state

    current_update_dict = _build_current_update_dict(
        trained_state_dict=trained_state_dict,
        global_state_dict=global_state_dict,
    )
    poisoned_update_dict = fedimp_attack_update(
        current_update_dict=current_update_dict,
        simulated_updates=simulated_updates,
        fedimp_factor=fedimp_factor,
        top_k_ratio=top_k_ratio,
        fisher_info=fisher_info,
        simulated_update_stats=simulated_update_stats,
    )

    poisoned_state_dict: Dict[str, Tensor] = {}
    for key, val in trained_state_dict.items():
        if key in poisoned_update_dict:
            if key not in global_state_dict:
                raise KeyError(f"global_state_dict 缺少 key: {key}")
            global_param = global_state_dict[key]
            if not torch.is_tensor(global_param):
                raise TypeError(f"global_state_dict['{key}'] 不是 Tensor。")
            poisoned_state_dict[key] = global_param + poisoned_update_dict[key].to(
                device=global_param.device,
                dtype=global_param.dtype,
            )
        else:
            poisoned_state_dict[key] = val.clone() if torch.is_tensor(val) else val

    return poisoned_state_dict


if __name__ == "__main__":
    torch.manual_seed(7)

    g = {
        "layer.weight": torch.ones(4, dtype=torch.float32),
        "layer.bias": torch.zeros(2, dtype=torch.float32),
    }
    local = {
        "layer.weight": g["layer.weight"] + torch.tensor([0.2, -0.1, 0.3, -0.4]),
        "layer.bias": g["layer.bias"] + torch.tensor([0.05, -0.02]),
    }

    sim_updates = [
        {
            "layer.weight": torch.tensor([0.15, -0.08, 0.25, -0.20]),
            "layer.bias": torch.tensor([0.03, -0.01]),
        },
        {
            "layer.weight": torch.tensor([0.18, -0.10, 0.28, -0.22]),
            "layer.bias": torch.tensor([0.04, -0.02]),
        },
    ]

    out = fedimp_attack(
        trained_state_dict=local,
        global_state_dict=g,
        simulated_updates=sim_updates,
        fedimp_factor=2.0,
        top_k_ratio=0.5,
    )
    print("global  :", g)
    print("trained :", local)
    print("poisoned:", out)
