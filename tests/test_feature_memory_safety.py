from __future__ import annotations

import unittest

import numpy as np
import torch

from aggregators.common import aggregate_geometric_median, geometric_median
from features import FeatureBuilder
from utils.heatmaps import pairwise_cosine_similarity
from utils.state_dict import flatten_tensor_dict, reconstruct_state_dict_like


class FeatureMemorySafetyTestCase(unittest.TestCase):
    def _build_local_states(self):
        return [
            {
                "layer.weight": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
                "counter": torch.tensor(1, dtype=torch.int64),
            },
            {
                "layer.weight": torch.tensor([2.0, 1.0, 0.0, -1.0], dtype=torch.float32),
                "counter": torch.tensor(2, dtype=torch.int64),
            },
            {
                "layer.weight": torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32),
                "counter": torch.tensor(3, dtype=torch.int64),
            },
        ]

    def test_pairwise_embedding_fallback_preserves_geometry(self) -> None:
        local_states = self._build_local_states()
        builder = FeatureBuilder(
            model_name="toy",
            key_layer_map={"toy": ["layer"]},
            control_layer_map={"toy": ["layer"]},
            projection_dim=2,
            projection_seed=123,
            feature_chunk_size=2,
            max_dense_feature_bytes=1,
            max_projection_matrix_bytes=1,
        )

        feature_set = builder.build_feature_set(local_states, "selected_layers")
        dense_matrix = torch.stack(
            [flatten_tensor_dict(local_state, keys=["layer.weight"]) for local_state in local_states],
            dim=0,
        )

        self.assertEqual(feature_set.storage_mode, "pairwise_embedded")
        expected_distances = torch.cdist(dense_matrix, dense_matrix, p=2).pow(2)
        actual_distances = torch.cdist(feature_set.aggregator_matrix, feature_set.aggregator_matrix, p=2).pow(2)
        self.assertTrue(torch.allclose(actual_distances, expected_distances, atol=1e-5, rtol=1e-5))
        np.testing.assert_allclose(
            feature_set.cosine_similarity_matrix,
            pairwise_cosine_similarity(dense_matrix),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_hashed_projection_path_is_deterministic(self) -> None:
        local_states = self._build_local_states()
        builder = FeatureBuilder(
            model_name="toy",
            key_layer_map={"toy": ["layer"]},
            control_layer_map={"toy": ["layer"]},
            projection_dim=3,
            projection_seed=456,
            feature_chunk_size=2,
            max_dense_feature_bytes=1,
            max_projection_matrix_bytes=1,
        )

        first = builder.build_feature_set(local_states, "selected_layers_projected")
        second = builder.build_feature_set(local_states, "selected_layers_projected")

        self.assertEqual(first.storage_mode, "hashed_projected")
        self.assertEqual(tuple(first.aggregator_matrix.shape), (3, 3))
        self.assertTrue(torch.allclose(first.aggregator_matrix, second.aggregator_matrix))

    def test_selected_layers_balanced_scales_each_prefix_by_sqrt_dim(self) -> None:
        local_states = [
            {
                "front.weight": torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float32),
                "back.weight": torch.tensor([9.0], dtype=torch.float32),
            },
            {
                "front.weight": torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float32),
                "back.weight": torch.tensor([6.0], dtype=torch.float32),
            },
        ]
        builder = FeatureBuilder(
            model_name="toy",
            key_layer_map={"toy": ["front", "back"]},
            control_layer_map={"toy": ["front"]},
            projection_dim=2,
            projection_seed=321,
        )

        feature_set = builder.build_feature_set(local_states, "selected_layers_balanced")

        expected = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 9.0],
                [0.5, 1.5, 2.5, 3.5, 6.0],
            ],
            dtype=torch.float32,
        )
        self.assertEqual(feature_set.storage_mode, "dense_balanced")
        self.assertTrue(torch.allclose(feature_set.aggregator_matrix, expected))

    def test_balanced_projected_dense_path_matches_manual_projection(self) -> None:
        local_states = [
            {
                "front.weight": torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float32),
                "back.weight": torch.tensor([9.0], dtype=torch.float32),
            },
            {
                "front.weight": torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float32),
                "back.weight": torch.tensor([6.0], dtype=torch.float32),
            },
        ]
        builder = FeatureBuilder(
            model_name="toy",
            key_layer_map={"toy": ["front", "back"]},
            control_layer_map={"toy": ["front"]},
            projection_dim=3,
            projection_seed=789,
        )

        balanced_feature_set = builder.build_feature_set(local_states, "selected_layers_balanced")
        projected_feature_set = builder.build_feature_set(local_states, "selected_layers_balanced_projected")
        projection = builder._get_projection(balanced_feature_set.aggregator_matrix.size(1))

        self.assertEqual(projected_feature_set.storage_mode, "dense_balanced_projected")
        self.assertTrue(
            torch.allclose(
                projected_feature_set.aggregator_matrix,
                balanced_feature_set.aggregator_matrix @ projection,
            )
        )

    def test_streaming_geometric_median_matches_dense_reference(self) -> None:
        local_states = [
            {
                "weight": torch.tensor([0.0, 1.0], dtype=torch.float32),
                "bias": torch.tensor([0.0], dtype=torch.float32),
            },
            {
                "weight": torch.tensor([2.0, 1.0], dtype=torch.float32),
                "bias": torch.tensor([1.0], dtype=torch.float32),
            },
            {
                "weight": torch.tensor([1.0, 3.0], dtype=torch.float32),
                "bias": torch.tensor([2.0], dtype=torch.float32),
            },
        ]

        streaming_result = aggregate_geometric_median(
            local_state_dicts=local_states,
            reference_state_dict=local_states[0],
            max_iters=200,
            tol=1e-7,
        )

        dense_matrix = torch.stack([flatten_tensor_dict(local_state) for local_state in local_states], dim=0)
        dense_median = geometric_median(dense_matrix, max_iters=200, tol=1e-7)
        dense_result = reconstruct_state_dict_like(dense_median, local_states[0])

        self.assertTrue(torch.allclose(streaming_result["weight"], dense_result["weight"], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(streaming_result["bias"], dense_result["bias"], atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
