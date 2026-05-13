from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from utils.state_dict import (
    flatten_tensor_dict,
    get_float_tensor_keys,
    select_tensor_dict_by_prefixes,
)

from .projection import (
    build_hash_projection_plan,
    build_orthogonal_projection,
    stable_string_seed,
)


@dataclass
class BuiltFeatureSet:
    aggregator_matrix: torch.Tensor
    cosine_similarity_matrix: Optional[np.ndarray]
    feature_dim: int
    storage_mode: str


class FeatureBuilder:
    def __init__(
        self,
        model_name: str,
        key_layer_map: Dict[str, List[str]],
        control_layer_map: Dict[str, List[str]],
        projection_dim: int,
        projection_seed: int,
        feature_chunk_size: int = 65536,
        max_dense_feature_bytes: Optional[int] = None,
        max_projection_matrix_bytes: Optional[int] = None,
        compute_device: str = "cpu",
        max_gpu_feature_bytes: Optional[int] = None,
        balanced_extra_layer_map: Optional[Dict[str, List[str]]] = None,
        include_batch_norm_in_balanced: bool = False,
    ) -> None:
        self.model_name = str(model_name).strip().lower()
        self.key_layer_map = key_layer_map
        self.control_layer_map = control_layer_map
        self.balanced_extra_layer_map = balanced_extra_layer_map or {}
        self.include_batch_norm_in_balanced = bool(include_batch_norm_in_balanced)
        self.projection_dim = int(projection_dim)
        self.projection_seed = int(projection_seed)
        self.feature_chunk_size = int(feature_chunk_size)
        self.max_dense_feature_bytes = max_dense_feature_bytes
        self.max_projection_matrix_bytes = max_projection_matrix_bytes
        self.compute_device = self._resolve_compute_device(compute_device)
        self.max_gpu_feature_bytes = max_gpu_feature_bytes
        self._projection_cache: Dict[tuple[int, str], torch.Tensor] = {}
        self._hash_projection_plan_cache: Dict[
            tuple[str, int, int, int, str],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._feature_layout_cache: Dict[
            str,
            tuple[List[str], int, Optional[Dict[str, float]]],
        ] = {}

    def _resolve_compute_device(self, compute_device: str) -> torch.device:
        normalized_device = str(compute_device).strip().lower()
        if normalized_device in {"auto", "cuda"} and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def get_selected_prefixes(self) -> List[str]:
        prefixes = self.key_layer_map.get(self.model_name)
        if not prefixes:
            raise KeyError(f"No key layers configured for model: {self.model_name}")
        return list(prefixes)

    def get_control_prefixes(self) -> List[str]:
        prefixes = self.control_layer_map.get(self.model_name)
        if not prefixes:
            raise KeyError(f"No control layers configured for model: {self.model_name}")
        return list(prefixes)

    def get_balanced_extra_prefixes(self) -> List[str]:
        prefixes = self.balanced_extra_layer_map.get(self.model_name, [])
        return list(prefixes)

    def _dedupe_prefixes(self, prefixes: List[str]) -> List[str]:
        deduped: List[str] = []
        seen = set()
        for prefix in prefixes:
            normalized = str(prefix).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _has_float_key_with_prefix(self, reference_state, prefix: str) -> bool:
        for key, value in reference_state.items():
            if not torch.is_tensor(value) or not value.dtype.is_floating_point:
                continue
            if key == prefix or key.startswith(prefix + "."):
                return True
        return False

    def _infer_all_batch_norm_prefixes(self, reference_state) -> List[str]:
        inferred_prefixes: List[str] = []
        for key, value in reference_state.items():
            if not torch.is_tensor(value) or not value.dtype.is_floating_point:
                continue
            if not (key.endswith(".running_mean") or key.endswith(".running_var")):
                continue

            # BN 层都有 running_mean / running_var；用统计量反推前缀，
            # 可以覆盖 stem、残差块和 downsample 中所有命名形式的 BN。
            prefix = key.rsplit(".", 1)[0]
            if self._has_float_key_with_prefix(reference_state, prefix):
                inferred_prefixes.append(prefix)
        return self._dedupe_prefixes(inferred_prefixes)

    def _get_selected_prefixes_for_mode(self, reference_state, feature_mode: str) -> List[str]:
        selected_prefixes = self.get_selected_prefixes()
        normalized_mode = str(feature_mode).strip().lower()
        if normalized_mode not in {
            "selected_layers_balanced",
            "selected_layers_balanced_projected",
        }:
            return selected_prefixes

        balanced_prefixes = list(selected_prefixes)
        balanced_prefixes.extend(self.get_balanced_extra_prefixes())
        if self.include_batch_norm_in_balanced:
            # balanced 模式下把当前模型的所有 BN 作为独立层块加入；
            # 后续仍按各自维度做 sqrt(dim) 缩放，避免 BN 的小维度被大卷积淹没。
            balanced_prefixes.extend(self._infer_all_batch_norm_prefixes(reference_state))
        return self._dedupe_prefixes(balanced_prefixes)

    def _build_raw_matrix(self, local_state_dicts) -> torch.Tensor:
        rows = [flatten_tensor_dict(local_state_dict) for local_state_dict in local_state_dicts]
        return torch.stack(rows, dim=0)

    def _build_selected_matrix(self, local_state_dicts) -> torch.Tensor:
        prefixes = self.get_selected_prefixes()
        rows = []
        for local_state_dict in local_state_dicts:
            selected = select_tensor_dict_by_prefixes(local_state_dict, prefixes)
            if not selected:
                raise ValueError(
                    f"No parameters matched prefixes {prefixes} for model {self.model_name}."
                )
            rows.append(flatten_tensor_dict(selected))
        return torch.stack(rows, dim=0)

    def _build_control_matrix(self, local_state_dicts) -> torch.Tensor:
        prefixes = self.get_control_prefixes()
        rows = []
        for local_state_dict in local_state_dicts:
            selected = select_tensor_dict_by_prefixes(local_state_dict, prefixes)
            if not selected:
                raise ValueError(
                    f"No parameters matched control prefixes {prefixes} for model {self.model_name}."
                )
            rows.append(flatten_tensor_dict(selected))
        return torch.stack(rows, dim=0)

    def _resolve_matrix_device(self, estimated_bytes: int) -> torch.device:
        if self.compute_device.type != "cuda":
            return torch.device("cpu")
        if (
            self.max_gpu_feature_bytes is not None
            and int(estimated_bytes) > int(self.max_gpu_feature_bytes)
        ):
            return torch.device("cpu")
        return self.compute_device

    def _get_projection(self, input_dim: int, device: Optional[torch.device] = None) -> torch.Tensor:
        target_device = torch.device("cpu") if device is None else torch.device(device)
        cache_key = (int(input_dim), str(target_device))
        if cache_key not in self._projection_cache:
            projection = build_orthogonal_projection(
                input_dim=input_dim,
                output_dim=self.projection_dim,
                seed=self.projection_seed + input_dim,
            )
            self._projection_cache[cache_key] = projection.to(device=target_device)
        return self._projection_cache[cache_key]

    def _build_key_scale_map_for_prefixes(
        self,
        reference_state,
        prefixes: List[str],
    ) -> Dict[str, float]:
        key_scales: Dict[str, float] = {}
        for prefix in prefixes:
            selected = select_tensor_dict_by_prefixes(reference_state, [prefix])
            if not selected:
                raise ValueError(
                    f"feature_mode=selected_layers_balanced 在模型 {self.model_name} 上未匹配到前缀 {prefix}。"
                )
            layer_dim = int(sum(int(tensor.numel()) for tensor in selected.values()))
            if layer_dim <= 0:
                raise ValueError(f"前缀 {prefix} 对应的参数维度必须为正数。")

            # 按层块统一除以 sqrt(dim)，让大层不会仅凭维度优势主导距离。
            layer_scale = float(1.0 / math.sqrt(layer_dim))
            for key in selected.keys():
                key_scales[key] = layer_scale
        return key_scales

    def _resolve_feature_layout(
        self,
        local_state_dicts,
        feature_mode: str,
    ) -> tuple[List[str], int, Optional[Dict[str, float]]]:
        if not local_state_dicts:
            raise ValueError("local_state_dicts 不能为空。")

        normalized_mode = str(feature_mode).strip().lower()
        cached_layout = self._feature_layout_cache.get(normalized_mode)
        if cached_layout is not None:
            cached_keys, cached_feature_dim, cached_key_scales = cached_layout
            return (
                list(cached_keys),
                int(cached_feature_dim),
                dict(cached_key_scales) if cached_key_scales is not None else None,
            )

        reference_state = local_state_dicts[0]
        key_scales: Optional[Dict[str, float]] = None
        if normalized_mode == "raw_full":
            keys = get_float_tensor_keys(reference_state)
        elif normalized_mode in {
            "selected_layers",
            "selected_layers_projected",
            "selected_layers_balanced",
            "selected_layers_balanced_projected",
        }:
            prefixes = self._get_selected_prefixes_for_mode(reference_state, normalized_mode)
            keys = list(select_tensor_dict_by_prefixes(reference_state, prefixes).keys())
            if normalized_mode in {
                "selected_layers_balanced",
                "selected_layers_balanced_projected",
            }:
                key_scales = self._build_key_scale_map_for_prefixes(
                    reference_state=reference_state,
                    prefixes=prefixes,
                )
        elif normalized_mode == "control_layer":
            keys = list(
                select_tensor_dict_by_prefixes(reference_state, self.get_control_prefixes()).keys()
            )
        else:
            raise ValueError(f"Unsupported feature mode: {feature_mode}")

        if not keys:
            raise ValueError(f"feature_mode={feature_mode} 未匹配到任何浮点参数。")
        feature_dim = int(sum(int(reference_state[key].numel()) for key in keys))
        # 模型结构在同一轮实验中不变；缓存 key 列表和逐层缩放，
        # 避免每轮为 LazyStateDeltaSequence 重建完整 reference delta。
        self._feature_layout_cache[normalized_mode] = (
            list(keys),
            feature_dim,
            dict(key_scales) if key_scales is not None else None,
        )
        return keys, feature_dim, key_scales

    def _estimate_dense_matrix_bytes(
        self,
        num_clients: int,
        feature_dim: int,
    ) -> int:
        return int(num_clients) * int(feature_dim) * 4

    def _estimate_projection_matrix_bytes(
        self,
        input_dim: int,
        output_dim: int,
    ) -> int:
        return int(input_dim) * int(output_dim) * 4

    def _build_dense_matrix_from_keys(
        self,
        local_state_dicts,
        keys: List[str],
        key_scales: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        num_clients = len(local_state_dicts)
        if num_clients <= 0:
            raise ValueError("local_state_dicts 不能为空。")

        reference_state = local_state_dicts[0]
        key_sizes = [int(reference_state[key].numel()) for key in keys]
        feature_dim = int(sum(key_sizes))
        target_device = torch.device("cpu") if device is None else torch.device(device)

        dense_update_builder = getattr(local_state_dicts, "build_dense_update_matrix", None)
        if key_scales is None and callable(dense_update_builder):
            dense_payload = dense_update_builder(
                max_numel=max(1, num_clients * feature_dim),
                device=target_device,
            )
            if dense_payload is not None:
                matrix, dense_keys, _, _ = dense_payload
                if list(dense_keys) == list(keys):
                    # raw_full 的 delta 序列可直接写成 dense 矩阵，省掉逐客户端生成完整 delta dict。
                    return matrix

        matrix = torch.empty((num_clients, feature_dim), dtype=torch.float32, device=target_device)

        # 预分配整块矩阵并按切片填充，避免每个客户端先 cat、最后再 stack 的两次大额中间拷贝。
        for row_index, local_state_dict in enumerate(local_state_dicts):
            cursor = 0
            for key, key_size in zip(keys, key_sizes):
                flat_view = local_state_dict[key].detach().to(
                    device=target_device,
                    dtype=torch.float32,
                    non_blocking=target_device.type == "cuda",
                ).reshape(-1)
                if int(flat_view.numel()) != key_size:
                    raise ValueError(f"{key} 的参数维度在客户端之间不一致。")
                target_slice = matrix[row_index, cursor : cursor + key_size]
                scale = float(key_scales.get(key, 1.0)) if key_scales is not None else 1.0
                if scale == 1.0:
                    target_slice.copy_(flat_view)
                else:
                    target_slice.copy_(flat_view)
                    target_slice.mul_(scale)
                cursor += key_size
        return matrix

    def _iter_chunk_rows(
        self,
        local_state_dicts,
        keys: List[str],
        key_scales: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
    ):
        target_device = torch.device("cpu") if device is None else torch.device(device)
        for key in keys:
            flat_views = [
                local_state_dict[key].detach().to(
                    device=target_device,
                    dtype=torch.float32,
                    non_blocking=target_device.type == "cuda",
                ).reshape(-1)
                for local_state_dict in local_state_dicts
            ]
            numel = int(flat_views[0].numel())
            for start in range(0, numel, self.feature_chunk_size):
                end = min(numel, start + self.feature_chunk_size)
                chunk_rows = torch.stack([flat_view[start:end] for flat_view in flat_views], dim=0)
                if key_scales is not None:
                    # 流式路径里同样应用逐层缩放，保证大模型和稠密路径语义一致。
                    chunk_rows = chunk_rows * float(key_scales.get(key, 1.0))
                yield key, start, chunk_rows

    def _build_pairwise_stats_from_keys(
        self,
        local_state_dicts,
        keys: List[str],
        key_scales: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, np.ndarray]:
        num_clients = len(local_state_dicts)
        target_device = torch.device("cpu") if device is None else torch.device(device)
        dot_products = torch.zeros((num_clients, num_clients), dtype=torch.float64, device=target_device)
        squared_norms = torch.zeros(num_clients, dtype=torch.float64, device=target_device)

        # 当显式特征矩阵过大时，这里直接在分块上累计点积 / 范数；
        # 后续再从距离矩阵恢复一个低维等距嵌入，避免复制出数 GB 的中间张量；CUDA 可用时分块矩阵乘放到 GPU。
        for _, _, chunk_rows in self._iter_chunk_rows(
            local_state_dicts,
            keys,
            key_scales=key_scales,
            device=target_device,
        ):
            chunk_rows_64 = chunk_rows.to(dtype=torch.float64)
            dot_products += chunk_rows_64 @ chunk_rows_64.t()
            squared_norms += (chunk_rows_64 * chunk_rows_64).sum(dim=1)

        squared_distances = (
            squared_norms.unsqueeze(1) + squared_norms.unsqueeze(0) - 2.0 * dot_products
        ).clamp_min(0.0)
        norm_products = torch.sqrt(squared_norms.unsqueeze(1) * squared_norms.unsqueeze(0))
        cosine_similarity = torch.zeros_like(dot_products)
        valid_mask = norm_products > 1e-12
        cosine_similarity[valid_mask] = dot_products[valid_mask] / norm_products[valid_mask]
        cosine_similarity = cosine_similarity.clamp(min=-1.0, max=1.0)
        return squared_distances.to(dtype=torch.float32), cosine_similarity.to(dtype=torch.float32).cpu().numpy()

    def _embed_from_squared_distances(
        self,
        squared_distances: torch.Tensor,
    ) -> torch.Tensor:
        num_clients = int(squared_distances.size(0))
        if num_clients == 1:
            return torch.zeros((1, 1), dtype=torch.float32, device=squared_distances.device)

        distance_matrix = squared_distances.to(dtype=torch.float64)
        centering = torch.eye(num_clients, dtype=torch.float64, device=distance_matrix.device) - (
            torch.ones((num_clients, num_clients), dtype=torch.float64, device=distance_matrix.device) / float(num_clients)
        )
        gram_matrix = -0.5 * centering @ distance_matrix @ centering
        eigenvalues, eigenvectors = torch.linalg.eigh(gram_matrix)
        positive_mask = eigenvalues > 1e-10
        if not torch.any(positive_mask):
            return torch.zeros((num_clients, 1), dtype=torch.float32, device=distance_matrix.device)

        eigenvalues = eigenvalues[positive_mask]
        eigenvectors = eigenvectors[:, positive_mask]
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        embedding = eigenvectors * torch.sqrt(eigenvalues).unsqueeze(0)
        return embedding.to(dtype=torch.float32)

    def _build_projected_matrix_dense(
        self,
        local_state_dicts,
        keys: List[str],
        key_scales: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        selected_matrix = self._build_dense_matrix_from_keys(
            local_state_dicts,
            keys,
            key_scales=key_scales,
            device=device,
        )
        projection = self._get_projection(selected_matrix.size(1), device=selected_matrix.device)
        return selected_matrix @ projection

    def _get_hash_projection_plan(
        self,
        key: str,
        chunk_start: int,
        chunk_size: int,
        output_dim: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (
            str(key),
            int(chunk_start),
            int(chunk_size),
            int(output_dim),
            str(device),
        )
        if cache_key not in self._hash_projection_plan_cache:
            bucket_indices, signs = build_hash_projection_plan(
                chunk_start=chunk_start,
                chunk_size=chunk_size,
                output_dim=output_dim,
                seed=stable_string_seed(key, self.projection_seed),
            )
            # 同一模型/特征模式每轮都会重复访问相同 chunk；
            # 缓存设备侧映射，避免反复生成 bucket/sign 并做 CPU->GPU 拷贝。
            self._hash_projection_plan_cache[cache_key] = (
                bucket_indices.to(device=device),
                signs.to(device=device),
            )
        return self._hash_projection_plan_cache[cache_key]

    def _build_projected_matrix_hashed(
        self,
        local_state_dicts,
        keys: List[str],
        output_dim: int,
        key_scales: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        num_clients = len(local_state_dicts)
        target_device = torch.device("cpu") if device is None else torch.device(device)
        projected = torch.zeros((num_clients, int(output_dim)), dtype=torch.float32, device=target_device)

        # 大模型下这里改为 CountSketch 风格的流式投影：
        # 逐块把参数散列到低维桶中，避免构造超大的稠密投影矩阵；小投影累加可直接留在 GPU。
        for key, start, chunk_rows in self._iter_chunk_rows(
            local_state_dicts,
            keys,
            key_scales=key_scales,
            device=target_device,
        ):
            bucket_indices, signs = self._get_hash_projection_plan(
                key=key,
                chunk_start=start,
                chunk_size=int(chunk_rows.size(1)),
                output_dim=output_dim,
                device=target_device,
            )
            projected.index_add_(1, bucket_indices, chunk_rows * signs.unsqueeze(0))
        return projected

    def build_feature_set(
        self,
        local_state_dicts,
        feature_mode: str,
    ) -> BuiltFeatureSet:
        normalized_mode = str(feature_mode).strip().lower()
        keys, feature_dim, key_scales = self._resolve_feature_layout(local_state_dicts, normalized_mode)
        num_clients = len(local_state_dicts)
        is_balanced_mode = normalized_mode in {
            "selected_layers_balanced",
            "selected_layers_balanced_projected",
        }
        is_projected_mode = normalized_mode in {
            "selected_layers_projected",
            "selected_layers_balanced_projected",
        }

        if is_projected_mode:
            output_dim = min(int(self.projection_dim), int(feature_dim))
            dense_feature_bytes = self._estimate_dense_matrix_bytes(num_clients=num_clients, feature_dim=feature_dim)
            projection_bytes = self._estimate_projection_matrix_bytes(
                input_dim=feature_dim,
                output_dim=output_dim,
            )
            can_materialize_dense = (
                self.max_dense_feature_bytes is None
                or dense_feature_bytes <= int(self.max_dense_feature_bytes)
            )
            can_materialize_projection = (
                self.max_projection_matrix_bytes is None
                or projection_bytes <= int(self.max_projection_matrix_bytes)
            )
            if can_materialize_dense and can_materialize_projection:
                matrix_device = self._resolve_matrix_device(max(dense_feature_bytes, projection_bytes))
                aggregator_matrix = self._build_projected_matrix_dense(
                    local_state_dicts,
                    keys,
                    key_scales=key_scales,
                    device=matrix_device,
                )
                return BuiltFeatureSet(
                    aggregator_matrix=aggregator_matrix,
                    cosine_similarity_matrix=None,
                    feature_dim=int(aggregator_matrix.size(1)),
                    storage_mode="dense_balanced_projected" if is_balanced_mode else "dense_projected",
                )

            matrix_device = self._resolve_matrix_device(dense_feature_bytes)
            aggregator_matrix = self._build_projected_matrix_hashed(
                local_state_dicts=local_state_dicts,
                keys=keys,
                output_dim=output_dim,
                key_scales=key_scales,
                device=matrix_device,
            )
            return BuiltFeatureSet(
                aggregator_matrix=aggregator_matrix,
                cosine_similarity_matrix=None,
                feature_dim=int(aggregator_matrix.size(1)),
                storage_mode="hashed_balanced_projected" if is_balanced_mode else "hashed_projected",
            )

        dense_feature_bytes = self._estimate_dense_matrix_bytes(num_clients=num_clients, feature_dim=feature_dim)
        if self.max_dense_feature_bytes is None or dense_feature_bytes <= int(self.max_dense_feature_bytes):
            matrix_device = self._resolve_matrix_device(dense_feature_bytes)
            aggregator_matrix = self._build_dense_matrix_from_keys(
                local_state_dicts,
                keys,
                key_scales=key_scales,
                device=matrix_device,
            )
            return BuiltFeatureSet(
                aggregator_matrix=aggregator_matrix,
                cosine_similarity_matrix=None,
                feature_dim=int(aggregator_matrix.size(1)),
                storage_mode="dense_balanced" if is_balanced_mode else "dense",
            )

        matrix_device = self._resolve_matrix_device(dense_feature_bytes)
        squared_distances, cosine_similarity_matrix = self._build_pairwise_stats_from_keys(
            local_state_dicts=local_state_dicts,
            keys=keys,
            key_scales=key_scales,
            device=matrix_device,
        )
        aggregator_matrix = self._embed_from_squared_distances(squared_distances)
        return BuiltFeatureSet(
            aggregator_matrix=aggregator_matrix,
            cosine_similarity_matrix=cosine_similarity_matrix,
            feature_dim=feature_dim,
            storage_mode="pairwise_embedded_balanced" if is_balanced_mode else "pairwise_embedded",
        )

    def build_feature_matrix(
        self,
        local_state_dicts,
        feature_mode: str,
    ) -> torch.Tensor:
        return self.build_feature_set(
            local_state_dicts=local_state_dicts,
            feature_mode=feature_mode,
        ).aggregator_matrix
