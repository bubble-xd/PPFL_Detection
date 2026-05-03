"""
ALIE 攻击实现。

参考:
    A Little Is Enough: Circumventing Defenses For Distributed Learning
    https://proceedings.neurips.cc/paper_files/paper/2019/hash/ec1c59141046cd1866bbbcdfb6ae31d4-Abstract.html
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

try:
    from scipy.stats import norm as _norm
except Exception:
    _norm = None


_FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)


@dataclass
class ALIEUpdateStats:
    """良性客户端 update 的逐层均值/标准差缓存。"""

    mean: Dict[str, Tensor]
    std: Dict[str, Tensor]
    count: Dict[str, int]


def _normal_ppf(value: float) -> float:
    if not (0.0 < value < 1.0):
        raise ValueError("value 必须在 (0, 1) 区间。")

    if _norm is not None:
        return float(_norm.ppf(value))

    dist = torch.distributions.Normal(0.0, 1.0)
    return float(dist.icdf(torch.tensor(value)).item())


def _get_float_param_keys(state_dict: Dict[str, Tensor]) -> List[str]:
    keys: List[str] = []
    for key, val in state_dict.items():
        if torch.is_tensor(val) and val.dtype in _FLOAT_DTYPES:
            keys.append(key)
    return keys


def _resolve_z_max(num_clients: int, num_adv: int, z_max: Optional[float]) -> float:
    if z_max is not None:
        return float(z_max)

    s = torch.floor(torch.tensor(num_clients / 2 + 1.0)) - float(num_adv)
    denom = float(num_clients - num_adv)
    cdf_value = (num_clients - num_adv - float(s)) / denom

    # 防止数值边界导致 ppf(0/1) 返回 inf
    eps = 1e-8
    cdf_value = min(max(cdf_value, eps), 1.0 - eps)
    return _normal_ppf(float(cdf_value))


def _proxy_scale_stats(proxy_scales: Sequence[float]) -> Tuple[float, float]:
    if proxy_scales is None or len(proxy_scales) == 0:
        raise ValueError("proxy_scales 至少包含一个缩放系数。")

    scales = torch.tensor(list(proxy_scales), dtype=torch.float32)
    mean_scale = float(scales.mean().item())
    std_scale = float(scales.std(unbiased=False).item())
    return mean_scale, std_scale


def _derive_client_jitter_seed(client_id: int, key: str) -> int:
    payload = f"alie-client-jitter:{int(client_id)}:{key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _sample_client_jitter(
    reference_std: Tensor,
    client_id: int,
    key: str,
    client_jitter_std: float,
) -> Tensor:
    if client_jitter_std <= 0.0:
        return torch.zeros_like(reference_std)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_derive_client_jitter_seed(client_id=client_id, key=key))
    noise = torch.randn(
        reference_std.shape,
        generator=generator,
        device=reference_std.device,
        dtype=reference_std.dtype,
    )
    return noise * reference_std * float(client_jitter_std)


def _stream_state_update_stats_for_key(
    benign_state_dicts: Sequence[Dict[str, Tensor]],
    global_state_dict: Dict[str, Tensor],
    key: str,
) -> Tuple[Tensor, Tensor]:
    global_param = global_state_dict[key]
    if not torch.is_tensor(global_param):
        raise TypeError(f"global_state_dict['{key}'] 不是 Tensor。")

    global_cpu = global_param.detach().to(device="cpu", dtype=torch.float32)
    mean = torch.zeros_like(global_cpu)
    m2 = torch.zeros_like(global_cpu)
    count = 0

    for idx, benign_state in enumerate(benign_state_dicts):
        if not isinstance(benign_state, dict):
            raise TypeError(f"benign_state_dicts[{idx}] 不是 Dict[str, Tensor]。")
        if key not in benign_state:
            raise KeyError(f"benign_state_dicts[{idx}] 缺少 key: {key}")

        local_param = benign_state[key]
        if not torch.is_tensor(local_param):
            raise TypeError(f"benign_state_dicts[{idx}]['{key}'] 不是 Tensor。")
        if local_param.shape != global_param.shape:
            raise ValueError(
                f"benign_state_dicts[{idx}]['{key}'] 与 global_state_dict['{key}'] 形状不一致。"
            )

        update = local_param.detach().to(device="cpu", dtype=torch.float32) - global_cpu
        count += 1

        delta = update - mean
        mean = mean + delta / float(count)
        delta2 = update - mean
        m2 = m2 + delta * delta2

    if count == 0:
        raise ValueError(f"没有可用于 key={key} 的良性更新。")

    variance = m2 / float(count)
    variance.clamp_(min=0.0)
    std = torch.sqrt(variance)
    return mean, std


def build_alie_update_stats(
    benign_state_dicts: Sequence[Dict[str, Tensor]],
    global_state_dict: Dict[str, Tensor],
) -> ALIEUpdateStats:
    """
    一次性统计所有良性客户端 update 的均值/标准差。

    ALIE 的均值和标准差与恶意客户端 id 无关；缓存后同一轮多个恶意客户端只需叠加各自 jitter。
    """
    if benign_state_dicts is None or len(benign_state_dicts) == 0:
        raise ValueError("benign_state_dicts 至少包含一个 state_dict。")
    if not isinstance(global_state_dict, dict):
        raise TypeError("global_state_dict 必须是 Dict[str, Tensor]。")

    keys = _get_float_param_keys(global_state_dict)
    global_cpu_by_key = {
        key: global_state_dict[key].detach().to(device="cpu", dtype=torch.float32)
        for key in keys
    }
    means: Dict[str, Tensor] = {}
    m2_values: Dict[str, Tensor] = {}
    counts: Dict[str, int] = {}

    for state_index, benign_state in enumerate(benign_state_dicts):
        if not isinstance(benign_state, dict):
            raise TypeError(f"benign_state_dicts[{state_index}] 不是 Dict[str, Tensor]。")

        for key in keys:
            if key not in benign_state:
                raise KeyError(f"benign_state_dicts[{state_index}] 缺少 key: {key}")
            local_param = benign_state[key]
            global_param_cpu = global_cpu_by_key[key]
            if not torch.is_tensor(local_param):
                raise TypeError(f"{key} 对应的 local/global 参数必须是 Tensor。")
            if local_param.shape != global_param_cpu.shape:
                raise ValueError(
                    f"benign_state_dicts[{state_index}]['{key}'] 与 global_state_dict['{key}'] 形状不一致。"
                )

            update = (
                local_param.detach().to(device="cpu", dtype=torch.float32)
                - global_param_cpu
            )
            if key not in means:
                means[key] = torch.zeros_like(update)
                m2_values[key] = torch.zeros_like(update)
                counts[key] = 0

            counts[key] += 1
            count = float(counts[key])
            delta = update - means[key]
            means[key].add_(delta / count)
            delta2 = update - means[key]
            m2_values[key].add_(delta * delta2)

    std_values: Dict[str, Tensor] = {}
    for key, mean in means.items():
        variance = m2_values[key].div(float(counts[key])).clamp_(min=0.0)
        std_values[key] = torch.sqrt(variance)

    return ALIEUpdateStats(mean=means, std=std_values, count=counts)


def _apply_update_dict(
    global_state_dict: Dict[str, Tensor],
    update_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    poisoned_state: Dict[str, Tensor] = {}
    for key, val in global_state_dict.items():
        if key in update_dict:
            if not torch.is_tensor(val):
                raise TypeError(f"global_state_dict['{key}'] 不是 Tensor。")
            poisoned_state[key] = val + update_dict[key].to(device=val.device, dtype=val.dtype)
        else:
            poisoned_state[key] = val.clone() if torch.is_tensor(val) else val
    return poisoned_state


def alie_attack_update(
    benign_state_dicts: Sequence[Dict[str, Tensor]],
    global_state_dict: Dict[str, Tensor],
    num_clients: int,
    num_adv: int,
    z_max: Optional[float] = None,
    client_id: Optional[int] = None,
    client_jitter_std: float = 0.0,
    update_stats: Optional[ALIEUpdateStats] = None,
) -> Dict[str, Tensor]:
    """
    直接返回恶意上传更新，而不是先构造完整投毒 state_dict。

    这里按层流式统计均值/标准差，避免把整模型更新展平后再 stack，
    能显著降低大模型上的峰值内存占用。
    """
    if benign_state_dicts is None or len(benign_state_dicts) == 0:
        raise ValueError("benign_state_dicts 至少包含一个 state_dict。")
    if not isinstance(global_state_dict, dict):
        raise TypeError("global_state_dict 必须是 Dict[str, Tensor]。")
    if num_clients <= 0:
        raise ValueError("num_clients 必须大于 0。")
    if num_adv < 0:
        raise ValueError("num_adv 不能为负数。")
    if num_clients <= num_adv:
        raise ValueError("ALIE 要求 num_clients > num_adv。")
    if client_jitter_std < 0:
        raise ValueError("client_jitter_std 不能为负数。")
    if client_jitter_std > 0.0 and client_id is None:
        raise ValueError("client_jitter_std > 0 时必须提供 client_id。")

    keys = _get_float_param_keys(global_state_dict)
    if not keys:
        return {}

    z_value = _resolve_z_max(num_clients=num_clients, num_adv=num_adv, z_max=z_max)
    attack_update: Dict[str, Tensor] = {}
    for key in keys:
        if update_stats is not None and key in update_stats.mean:
            mean = update_stats.mean[key]
            std = update_stats.std[key]
            if mean.shape != global_state_dict[key].shape or std.shape != global_state_dict[key].shape:
                raise ValueError(f"update_stats['{key}'] 形状与 global_state_dict 不一致。")
        else:
            mean, std = _stream_state_update_stats_for_key(
                benign_state_dicts=benign_state_dicts,
                global_state_dict=global_state_dict,
                key=key,
            )
        update = mean + z_value * std
        if client_jitter_std > 0.0:
            update = update + _sample_client_jitter(
                reference_std=std,
                client_id=int(client_id),
                key=key,
                client_jitter_std=client_jitter_std,
            )
        attack_update[key] = update

    return attack_update


def alie_attack_update_from_proxy(
    current_update_dict: Dict[str, Tensor],
    num_clients: int,
    num_adv: int,
    proxy_scales: Sequence[float],
    noise_std: float = 0.0,
    z_max: Optional[float] = None,
    client_id: Optional[int] = None,
    client_jitter_std: float = 0.0,
) -> Dict[str, Tensor]:
    """
    基于代理缩放统计直接近似 ALIE 上传更新。

    默认配置下代理良性状态本质是:
        update_i = scale_i * update + noise_i
    因此无需真正构造多份完整 state_dict，也能直接得到逐元素均值与标准差。
    """
    if not isinstance(current_update_dict, dict):
        raise TypeError("current_update_dict 必须是 Dict[str, Tensor]。")
    if num_clients <= 0:
        raise ValueError("num_clients 必须大于 0。")
    if num_adv < 0:
        raise ValueError("num_adv 不能为负数。")
    if num_clients <= num_adv:
        raise ValueError("ALIE 要求 num_clients > num_adv。")
    if noise_std < 0:
        raise ValueError("noise_std 不能为负数。")
    if client_jitter_std < 0:
        raise ValueError("client_jitter_std 不能为负数。")
    if client_jitter_std > 0.0 and client_id is None:
        raise ValueError("client_jitter_std > 0 时必须提供 client_id。")

    mean_scale, std_scale = _proxy_scale_stats(proxy_scales)
    z_value = _resolve_z_max(num_clients=num_clients, num_adv=num_adv, z_max=z_max)
    attack_update: Dict[str, Tensor] = {}

    for key, update in current_update_dict.items():
        if not torch.is_tensor(update):
            raise TypeError(f"current_update_dict['{key}'] 不是 Tensor。")
        if update.dtype not in _FLOAT_DTYPES:
            continue

        update_cpu = update.detach().to(device="cpu", dtype=torch.float32)
        variance = update_cpu.pow(2) * (std_scale ** 2)
        if noise_std > 0.0:
            variance = variance + float(noise_std) ** 2
        variance.clamp_(min=0.0)
        std = torch.sqrt(variance)
        update = update_cpu * mean_scale + z_value * std
        if client_jitter_std > 0.0:
            update = update + _sample_client_jitter(
                reference_std=std,
                client_id=int(client_id),
                key=key,
                client_jitter_std=client_jitter_std,
            )
        attack_update[key] = update

    return attack_update


def alie_attack(
    benign_state_dicts: Sequence[Dict[str, Tensor]],
    global_state_dict: Dict[str, Tensor],
    num_clients: int,
    num_adv: int,
    z_max: Optional[float] = None,
) -> Dict[str, Tensor]:
    """
    ALIE 攻击: 基于良性更新的均值和标准差构造恶意更新。

    参数:
        benign_state_dicts: 良性客户端训练后的模型参数列表
        global_state_dict: 全局模型参数
        num_clients: 客户端总数
        num_adv: 恶意客户端数量
        z_max: 可选 z 值；不传则按论文公式计算

    返回:
        恶意客户端应上传的投毒后模型参数
    """
    attack_update = alie_attack_update(
        benign_state_dicts=benign_state_dicts,
        global_state_dict=global_state_dict,
        num_clients=num_clients,
        num_adv=num_adv,
        z_max=z_max,
    )
    return _apply_update_dict(global_state_dict, attack_update)


if __name__ == "__main__":
    torch.manual_seed(7)
    global_model = {
        "layer.weight": torch.ones(4, dtype=torch.float32),
        "layer.bias": torch.zeros(2, dtype=torch.float32),
    }
    benign_states = [
        {
            "layer.weight": global_model["layer.weight"] + torch.tensor([0.12, -0.08, 0.10, -0.05]),
            "layer.bias": global_model["layer.bias"] + torch.tensor([0.02, -0.01]),
        },
        {
            "layer.weight": global_model["layer.weight"] + torch.tensor([0.10, -0.07, 0.12, -0.06]),
            "layer.bias": global_model["layer.bias"] + torch.tensor([0.03, -0.01]),
        },
        {
            "layer.weight": global_model["layer.weight"] + torch.tensor([0.11, -0.09, 0.11, -0.07]),
            "layer.bias": global_model["layer.bias"] + torch.tensor([0.02, -0.02]),
        },
    ]

    poisoned = alie_attack(
        benign_state_dicts=benign_states,
        global_state_dict=global_model,
        num_clients=10,
        num_adv=2,
    )
    print("global  :", global_model)
    print("poisoned:", poisoned)
