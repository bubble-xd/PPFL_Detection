from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor


def get_float_tensor_keys(tensor_dict: Dict[str, Tensor]) -> List[str]:
    keys: List[str] = []
    for key, value in tensor_dict.items():
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            keys.append(key)
    return keys


def clone_state_dict(
    state_dict: Dict[str, Tensor],
    device: Optional[str] = None,
) -> Dict[str, Tensor]:
    cloned: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned_value = value.detach().clone()
            if device is not None:
                cloned_value = cloned_value.to(device)
            cloned[key] = cloned_value
        else:
            cloned[key] = value
    return cloned


def build_state_delta_dict(
    local_state_dict: Dict[str, Tensor],
    global_state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """构造 local/global 之间的浮点参数差值。"""
    delta_dict: Dict[str, Tensor] = {}
    for key, local_value in local_state_dict.items():
        if key not in global_state_dict:
            raise KeyError(f"global_state_dict 缺少 key: {key}")
        global_value = global_state_dict[key]
        if torch.is_tensor(local_value) and local_value.dtype.is_floating_point:
            delta_dict[key] = (
                local_value.detach().to(device="cpu", dtype=torch.float32)
                - global_value.detach().to(device="cpu", dtype=torch.float32)
            )
    return delta_dict


def average_tensor_dicts(
    tensor_dicts: List[Dict[str, Tensor]],
    selected_ids: Optional[Iterable[int]] = None,
    reference_state_dict: Optional[Dict[str, Tensor]] = None,
) -> Dict[str, Tensor]:
    if not tensor_dicts:
        raise ValueError("tensor_dicts must not be empty.")

    indices = list(selected_ids) if selected_ids is not None else list(range(len(tensor_dicts)))
    if not indices:
        raise ValueError("selected_ids must not be empty.")

    source_reference = tensor_dicts[indices[0]]
    output_reference = reference_state_dict if reference_state_dict is not None else source_reference
    float_keys = get_float_tensor_keys(source_reference)
    float_key_set = set(float_keys)
    # 这里既可以平均客户端本地模型，也可以平均 update。
    # 浮点字段后面会被均值覆盖，因此不先 clone 整份参考模型，避免大模型聚合时多做一遍完整拷贝。
    averaged: Dict[str, Tensor] = {}
    for key, value in output_reference.items():
        if key in float_key_set:
            continue
        averaged[key] = value.detach().clone() if torch.is_tensor(value) else value
    accumulators = {
        key: torch.zeros_like(source_reference[key], device="cpu", dtype=torch.float32)
        for key in float_keys
    }

    # 这里改成“按 state 顺序累加”：
    # 对磁盘后端来说，每个客户端 state 只需完整回读一次，避免按 key 外层循环反复读盘。
    for index in indices:
        state_dict = tensor_dicts[index]
        for key in float_keys:
            accumulators[key].add_(state_dict[key].detach().to(device="cpu", dtype=torch.float32))

    for key in float_keys:
        mean_tensor = accumulators[key].div_(float(len(indices)))
        reference_value = output_reference[key]
        if torch.is_tensor(reference_value):
            averaged[key] = mean_tensor.to(device=reference_value.device, dtype=reference_value.dtype)
        else:
            averaged[key] = mean_tensor
    return averaged


def flatten_tensor_dict(
    tensor_dict: Dict[str, Tensor],
    keys: Optional[Iterable[str]] = None,
) -> Tensor:
    selected_keys = list(keys) if keys is not None else get_float_tensor_keys(tensor_dict)
    if not selected_keys:
        return torch.empty(0, dtype=torch.float32)
    flat_chunks = [
        tensor_dict[key].detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        for key in selected_keys
    ]
    return torch.cat(flat_chunks, dim=0)


def reconstruct_state_dict_like(
    flat_tensor: Tensor,
    reference_state_dict: Dict[str, Tensor],
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, Tensor]:
    selected_keys = list(keys) if keys is not None else get_float_tensor_keys(reference_state_dict)
    reconstructed = clone_state_dict(reference_state_dict)
    cursor = 0
    for key in selected_keys:
        reference = reference_state_dict[key]
        if not torch.is_tensor(reference):
            raise TypeError(f"reference_state_dict['{key}'] 不是 Tensor。")
        size = int(reference.numel())
        reconstructed[key] = (
            flat_tensor[cursor : cursor + size]
            .reshape_as(reference)
            .to(device=reference.device, dtype=reference.dtype)
            .clone()
        )
        cursor += size
    if cursor != int(flat_tensor.numel()):
        raise ValueError("flat_tensor 的长度与 reference_state_dict 中的浮点参数数量不一致。")
    return reconstructed


def select_tensor_dict_by_prefixes(
    tensor_dict: Dict[str, Tensor],
    prefixes: Iterable[str],
) -> Dict[str, Tensor]:
    prefixes = tuple(prefixes)
    selected: Dict[str, Tensor] = {}
    for key, value in tensor_dict.items():
        if not torch.is_tensor(value) or not value.dtype.is_floating_point:
            continue
        if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes):
            selected[key] = value
    return selected
