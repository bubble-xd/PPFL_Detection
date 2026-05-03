from __future__ import annotations

import tempfile
import unittest

import torch

from aggregators.common import aggregate_mean
from attacks.untargeted.a_lie import alie_attack_update, build_alie_update_stats
from attacks.untargeted.fedimp import build_fedimp_simulated_update_stats, fedimp_attack
from config import Config
from features import FeatureBuilder
from utils.heatmaps import compute_cosine_group_metrics, pairwise_cosine_similarity
from utils.state_store import DiskStateStore, LazyStateDeltaSequence


class StateStoreTestCase(unittest.TestCase):
    def _build_state(self, weight_values):
        return {
            "layer.weight": torch.tensor(weight_values, dtype=torch.float32),
            "layer.bias": torch.tensor([0.5], dtype=torch.float32),
            "counter": torch.tensor(1, dtype=torch.int64),
        }

    def test_disk_state_store_view_works_with_feature_builder_and_aggregate_mean(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with DiskStateStore.create_temporary(parent_dir=temp_dir, prefix="state_test_") as store:
                store.save_state(1, self._build_state([3.0, 4.0]))
                store.save_state(0, self._build_state([1.0, 2.0]))

                ordered_states = store.build_view()
                feature_builder = FeatureBuilder(
                    model_name="toy",
                    key_layer_map={"toy": ["layer"]},
                    control_layer_map={"toy": ["layer"]},
                    projection_dim=2,
                    projection_seed=123,
                )
                feature_set = feature_builder.build_feature_set(ordered_states, "selected_layers")

                self.assertEqual(tuple(feature_set.aggregator_matrix.shape), (2, 3))
                self.assertTrue(
                    torch.allclose(
                        feature_set.aggregator_matrix,
                        torch.tensor([[1.0, 2.0, 0.5], [3.0, 4.0, 0.5]], dtype=torch.float32),
                    )
                )

                averaged = aggregate_mean(
                    local_state_dicts=ordered_states,
                    reference_state_dict=store.load_state(0),
                )
                self.assertTrue(
                    torch.allclose(
                        averaged["layer.weight"],
                        torch.tensor([2.0, 3.0], dtype=torch.float32),
                    )
                )

    def test_lazy_state_delta_sequence_supports_fedimp(self) -> None:
        global_state = {
            "layer.weight": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "layer.bias": torch.tensor([0.0], dtype=torch.float32),
        }
        benign_states = [
            {
                "layer.weight": torch.tensor([1.1, 0.9], dtype=torch.float32),
                "layer.bias": torch.tensor([0.1], dtype=torch.float32),
            },
            {
                "layer.weight": torch.tensor([0.8, 1.2], dtype=torch.float32),
                "layer.bias": torch.tensor([-0.1], dtype=torch.float32),
            },
        ]
        current_state = {
            "layer.weight": torch.tensor([1.4, 0.7], dtype=torch.float32),
            "layer.bias": torch.tensor([0.2], dtype=torch.float32),
        }

        poisoned_state = fedimp_attack(
            trained_state_dict=current_state,
            global_state_dict=global_state,
            simulated_updates=LazyStateDeltaSequence(benign_states, global_state),
            fedimp_factor=2.0,
            top_k_ratio=1.0,
        )

        self.assertEqual(set(poisoned_state.keys()), set(current_state.keys()))
        self.assertEqual(tuple(poisoned_state["layer.weight"].shape), (2,))

    def test_cached_fedimp_stats_match_direct_streaming(self) -> None:
        global_state = {
            "layer.weight": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            "layer.bias": torch.tensor([0.0], dtype=torch.float32),
        }
        benign_states = [
            {
                "layer.weight": torch.tensor([1.1, 0.9, 1.2], dtype=torch.float32),
                "layer.bias": torch.tensor([0.1], dtype=torch.float32),
            },
            {
                "layer.weight": torch.tensor([0.8, 1.2, 0.7], dtype=torch.float32),
                "layer.bias": torch.tensor([-0.1], dtype=torch.float32),
            },
            {
                "layer.weight": torch.tensor([1.3, 0.7, 1.1], dtype=torch.float32),
                "layer.bias": torch.tensor([0.2], dtype=torch.float32),
            },
        ]
        current_state = {
            "layer.weight": torch.tensor([1.4, 0.7, 1.5], dtype=torch.float32),
            "layer.bias": torch.tensor([0.2], dtype=torch.float32),
        }
        lazy_updates = LazyStateDeltaSequence(benign_states, global_state)
        cached_stats = build_fedimp_simulated_update_stats(lazy_updates)

        # 缓存统计量只改变计算路径，不应改变 FedImp 生成的恶意 state。
        direct_state = fedimp_attack(
            trained_state_dict=current_state,
            global_state_dict=global_state,
            simulated_updates=lazy_updates,
            fedimp_factor=2.0,
            top_k_ratio=0.5,
        )
        cached_state = fedimp_attack(
            trained_state_dict=current_state,
            global_state_dict=global_state,
            simulated_updates=lazy_updates,
            simulated_update_stats=cached_stats,
            fedimp_factor=2.0,
            top_k_ratio=0.5,
        )

        for key in current_state:
            self.assertTrue(torch.allclose(direct_state[key], cached_state[key]))

    def test_cached_alie_stats_match_direct_streaming(self) -> None:
        global_state = {
            "layer.weight": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            "layer.bias": torch.tensor([0.0], dtype=torch.float32),
        }
        benign_states = [
            {
                "layer.weight": torch.tensor([1.1, 0.9, 1.2], dtype=torch.float32),
                "layer.bias": torch.tensor([0.1], dtype=torch.float32),
            },
            {
                "layer.weight": torch.tensor([0.8, 1.2, 0.7], dtype=torch.float32),
                "layer.bias": torch.tensor([-0.1], dtype=torch.float32),
            },
            {
                "layer.weight": torch.tensor([1.3, 0.7, 1.1], dtype=torch.float32),
                "layer.bias": torch.tensor([0.2], dtype=torch.float32),
            },
        ]
        cached_stats = build_alie_update_stats(benign_states, global_state)

        # ALIE 的良性分布缓存不应改变攻击 update，只减少同一轮内的重复统计。
        direct_update = alie_attack_update(
            benign_state_dicts=benign_states,
            global_state_dict=global_state,
            num_clients=10,
            num_adv=3,
            z_max=1.5,
            client_id=1,
            client_jitter_std=0.0,
        )
        cached_update = alie_attack_update(
            benign_state_dicts=benign_states,
            global_state_dict=global_state,
            num_clients=10,
            num_adv=3,
            z_max=1.5,
            client_id=1,
            client_jitter_std=0.0,
            update_stats=cached_stats,
        )

        for key in direct_update:
            self.assertTrue(torch.allclose(direct_update[key], cached_update[key]))

    def test_lazy_state_delta_sequence_keeps_bm_gap_in_update_space(self) -> None:
        global_state = {
            "layer.weight": torch.tensor([1000.0, 0.0], dtype=torch.float32),
        }
        full_states = [
            {"layer.weight": torch.tensor([1000.0, 1.0], dtype=torch.float32)},
            {"layer.weight": torch.tensor([1000.0, 1.1], dtype=torch.float32)},
            {"layer.weight": torch.tensor([1000.0, -1.0], dtype=torch.float32)},
            {"layer.weight": torch.tensor([1000.0, -1.1], dtype=torch.float32)},
        ]
        feature_builder = FeatureBuilder(
            model_name="toy",
            key_layer_map={"toy": ["layer"]},
            control_layer_map={"toy": ["layer"]},
            projection_dim=2,
            projection_seed=123,
        )

        full_feature_set = feature_builder.build_feature_set(full_states, "selected_layers")
        delta_feature_set = feature_builder.build_feature_set(
            LazyStateDeltaSequence(full_states, global_state),
            "selected_layers",
        )

        # 共享全局权重会把完整模型的余弦相似度压到几乎完全一致，
        # 但在 delta 空间里，良/恶更新方向仍应保持明显可分。
        full_metrics = compute_cosine_group_metrics(
            similarity_matrix=pairwise_cosine_similarity(full_feature_set.aggregator_matrix),
            client_order=[0, 1, 2, 3],
            malicious_ids=[2, 3],
        )
        delta_metrics = compute_cosine_group_metrics(
            similarity_matrix=pairwise_cosine_similarity(delta_feature_set.aggregator_matrix),
            client_order=[0, 1, 2, 3],
            malicious_ids=[2, 3],
        )

        self.assertLess(abs(float(full_metrics["bm_gap"])), 1e-4)
        self.assertGreater(float(delta_metrics["bm_gap"]), 1.9)

    def test_only_vgg11_uses_disk_state_cache_by_default(self) -> None:
        self.assertTrue(Config.should_stream_client_states("vgg11"))
        self.assertFalse(Config.should_stream_client_states("lenet5"))
        self.assertFalse(Config.should_stream_client_states("resnet18"))


if __name__ == "__main__":
    unittest.main()
