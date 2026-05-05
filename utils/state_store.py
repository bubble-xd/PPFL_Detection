from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from typing import Dict, Optional

import torch
from torch import Tensor

from .state_dict import build_state_delta_dict


class StateSequenceView(Sequence[Dict[str, Tensor]]):
    """按给定客户端顺序惰性读取 state_dict，避免一次性常驻整轮客户端模型。"""

    def __init__(
        self,
        store: "DiskStateStore",
        client_ids,
    ) -> None:
        self.store = store
        self.client_ids = [int(client_id) for client_id in client_ids]

    def __len__(self) -> int:
        return len(self.client_ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.store.load_state(client_id) for client_id in self.client_ids[index]]
        return self.store.load_state(self.client_ids[int(index)])

    def __iter__(self) -> Iterator[Dict[str, Tensor]]:
        for client_id in self.client_ids:
            yield self.store.load_state(client_id)


class DiskStateStore(Sequence[Dict[str, Tensor]]):
    """
    round 级别的磁盘 state store。

    训练完每个客户端后立刻把本地模型写到磁盘，后续特征提取/攻击/聚合再按需回读，
    避免服务端同时常驻整轮所有完整 state_dict。
    """

    def __init__(
        self,
        cache_dir: str,
        cleanup_on_close: bool = True,
    ) -> None:
        self.cache_dir = str(cache_dir)
        self.cleanup_on_close = bool(cleanup_on_close)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._state_paths: Dict[int, str] = {}

    @classmethod
    def create_temporary(
        cls,
        parent_dir: str,
        prefix: str = ".round_state_cache_",
        cleanup_on_close: bool = True,
    ) -> "DiskStateStore":
        cache_dir = tempfile.mkdtemp(dir=str(parent_dir), prefix=str(prefix))
        return cls(cache_dir=cache_dir, cleanup_on_close=cleanup_on_close)

    def _build_state_path(self, client_id: int) -> str:
        return os.path.join(self.cache_dir, f"client_{int(client_id):03d}.pt")

    def save_state(
        self,
        client_id: int,
        state_dict: Dict[str, Tensor],
    ) -> str:
        path = self._build_state_path(client_id)
        torch.save(state_dict, path)
        self._state_paths[int(client_id)] = path
        return path

    def load_state(
        self,
        client_id: int,
    ) -> Dict[str, Tensor]:
        normalized_client_id = int(client_id)
        if normalized_client_id not in self._state_paths:
            raise KeyError(f"client_id={normalized_client_id} 对应的 state 尚未写入磁盘。")
        return torch.load(self._state_paths[normalized_client_id], map_location="cpu")

    def get_client_ids(self) -> list[int]:
        return sorted(int(client_id) for client_id in self._state_paths.keys())

    def build_view(
        self,
        client_ids: Optional[list[int]] = None,
    ) -> StateSequenceView:
        resolved_client_ids = self.get_client_ids() if client_ids is None else [int(client_id) for client_id in client_ids]
        return StateSequenceView(store=self, client_ids=resolved_client_ids)

    def __len__(self) -> int:
        return len(self._state_paths)

    def __getitem__(self, index):
        return self.build_view()[index]

    def __iter__(self) -> Iterator[Dict[str, Tensor]]:
        return iter(self.build_view())

    def close(self) -> None:
        if self.cleanup_on_close:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def __enter__(self) -> "DiskStateStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class LazyStateDeltaSequence(Sequence[Dict[str, Tensor]]):
    """惰性把本地 state 转成 update，供 FedImp 之类的攻击按需流式消费。"""

    def __init__(
        self,
        state_dicts: Sequence[Dict[str, Tensor]],
        global_state_dict: Dict[str, Tensor],
    ) -> None:
        self.state_dicts = state_dicts
        self.global_state_dict = global_state_dict

    def __len__(self) -> int:
        return len(self.state_dicts)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                build_state_delta_dict(state_dict, self.global_state_dict)
                for state_dict in self.state_dicts[index]
            ]
        return build_state_delta_dict(self.state_dicts[int(index)], self.global_state_dict)

    def __iter__(self) -> Iterator[Dict[str, Tensor]]:
        for state_dict in self.state_dicts:
            yield build_state_delta_dict(state_dict, self.global_state_dict)

    def get_float_metadata(self):
        update_count = len(self.state_dicts)
        if update_count <= 0:
            return [], {}, {}

        first_state = self.state_dicts[0]
        keys = [
            key
            for key, value in first_state.items()
            if torch.is_tensor(value) and value.dtype.is_floating_point
        ]
        shapes = {}
        sizes = {}
        for key in keys:
            if key not in self.global_state_dict:
                raise KeyError(f"global_state_dict 缺少 key: {key}")
            local_value = first_state[key]
            global_value = self.global_state_dict[key]
            if not torch.is_tensor(global_value):
                raise TypeError(f"global_state_dict['{key}'] 不是 Tensor。")
            if local_value.shape != global_value.shape:
                raise ValueError(f"{key} 的 local/global 形状不一致。")
            shapes[key] = local_value.shape
            sizes[key] = int(local_value.numel())
        return keys, shapes, sizes

    def build_update_sum_sumsq(
        self,
        device: str | torch.device = "cpu",
        keys=None,
        shapes=None,
    ):
        update_count = len(self.state_dicts)
        if update_count <= 0:
            return {}, {}, {}, {}

        if keys is None or shapes is None:
            keys, shapes, _ = self.get_float_metadata()

        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            target_device = torch.device("cpu")

        sums = {
            key: torch.zeros(shapes[key], dtype=torch.float32, device=target_device)
            for key in keys
        }
        sumsq_values = {
            key: torch.zeros(shapes[key], dtype=torch.float32, device=target_device)
            for key in keys
        }
        counts = {key: 0 for key in keys}
        scratch = {
            key: torch.empty(shapes[key], dtype=torch.float32, device=target_device)
            for key in keys
        }

        # FedImp 统计只需要 update 的一阶/二阶和；
        # 这里按客户端流式累加，避免先构造“全层 delta dict”再二次遍历。
        for row_index in range(update_count):
            state_dict = self.state_dicts[row_index]
            for key in keys:
                if key not in state_dict:
                    raise KeyError(f"state_dict 缺少 key: {key}")
                if key not in self.global_state_dict:
                    raise KeyError(f"global_state_dict 缺少 key: {key}")
                local_value = state_dict[key]
                global_value = self.global_state_dict[key]
                if not torch.is_tensor(local_value) or not local_value.dtype.is_floating_point:
                    raise TypeError(f"state_dict['{key}'] 不是浮点 Tensor。")
                if not torch.is_tensor(global_value):
                    raise TypeError(f"global_state_dict['{key}'] 不是 Tensor。")
                if local_value.shape != shapes[key] or global_value.shape != shapes[key]:
                    raise ValueError(f"{key} 的 local/global 形状不一致。")

                delta = scratch[key]
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
                    out=delta,
                )
                counts[key] += 1
                sums[key].add_(delta)
                sumsq_values[key].addcmul_(delta, delta)

        return sums, sumsq_values, counts, dict(shapes)

    def build_dense_update_matrix(self, max_numel: int, device: str | torch.device = "cpu"):
        update_count = len(self.state_dicts)
        if update_count <= 0:
            return None

        first_state = self.state_dicts[0]
        keys = [
            key
            for key, value in first_state.items()
            if torch.is_tensor(value) and value.dtype.is_floating_point
        ]
        if not keys:
            return torch.empty((update_count, 0), dtype=torch.float32), [], {}, {}

        shapes = {key: first_state[key].shape for key in keys}
        sizes = {key: int(first_state[key].numel()) for key in keys}
        total_numel = int(sum(sizes.values()))
        if total_numel * update_count > int(max_numel):
            return None

        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            target_device = torch.device("cpu")

        # FedImp 中小模型统计只需要一份 dense update 矩阵；
        # 直接把 local-global 写到目标切片，避免先生成每个客户端的完整 delta dict。
        matrix = torch.empty((update_count, total_numel), dtype=torch.float32, device=target_device)
        for row_index in range(update_count):
            state_dict = self.state_dicts[row_index]
            cursor = 0
            for key in keys:
                if key not in state_dict:
                    raise KeyError(f"state_dict 缺少 key: {key}")
                if key not in self.global_state_dict:
                    raise KeyError(f"global_state_dict 缺少 key: {key}")
                local_value = state_dict[key]
                global_value = self.global_state_dict[key]
                if not torch.is_tensor(local_value) or not local_value.dtype.is_floating_point:
                    raise TypeError(f"state_dict['{key}'] 不是浮点 Tensor。")
                if not torch.is_tensor(global_value):
                    raise TypeError(f"global_state_dict['{key}'] 不是 Tensor。")
                if local_value.shape != shapes[key] or global_value.shape != shapes[key]:
                    raise ValueError(f"{key} 的 local/global 形状不一致。")

                size = sizes[key]
                target = matrix[row_index, cursor : cursor + size].reshape(shapes[key])
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
                    out=target,
                )
                cursor += size
        return matrix, keys, shapes, sizes
