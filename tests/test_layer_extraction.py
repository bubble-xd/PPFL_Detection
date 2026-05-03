from __future__ import annotations

import unittest

import torch
from torch import nn

from layer_extraction.candidates import get_candidate_layer_prefixes
from layer_extraction.settings import LayerExtractionSettings
from layer_extraction.scoring import (
    compute_adaptive_weights,
    compute_population_variance,
    compute_round_layer_metrics,
    zscore_layer_values,
)
from layer_extraction.selection import select_layers
from layer_extraction.types import AttackLayerSummary
from models import build_model
from utils.state_dict import build_state_delta_dict


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3)
        self.bn = nn.BatchNorm2d(4)
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        raise NotImplementedError


class LayerExtractionTestCase(unittest.TestCase):
    def test_selected_models_use_fixed_dataset_mapping(self) -> None:
        settings = LayerExtractionSettings.from_config(
            selected_models=["lenet5", "resnet20", "vgg11"],
        )

        run_settings = settings.get_run_settings()

        self.assertEqual(
            [(item.get_model_name(), item.get_dataset_name()) for item in run_settings],
            [("lenet5", "mnist"), ("resnet20", "cifar10"), ("vgg11", "cifar10")],
        )

    def test_build_state_delta_dict_uses_float_params_only(self) -> None:
        local_state = {
            "weight": torch.tensor([2.0, 4.0], dtype=torch.float32),
            "counter": torch.tensor(5, dtype=torch.int64),
        }
        global_state = {
            "weight": torch.tensor([1.5, 3.0], dtype=torch.float32),
            "counter": torch.tensor(1, dtype=torch.int64),
        }

        delta = build_state_delta_dict(local_state, global_state)

        self.assertEqual(set(delta.keys()), {"weight"})
        self.assertTrue(torch.allclose(delta["weight"], torch.tensor([0.5, 1.0])))

    def test_candidate_layers_only_keep_leaf_conv_and_linear(self) -> None:
        candidates = get_candidate_layer_prefixes(_ToyModel())
        self.assertEqual(candidates, ["conv", "head"])

    def test_resnet_candidates_skip_projection_shortcuts(self) -> None:
        expected_counts = {
            "resnet20": 20,
            "resnet18": 18,
            "resnet34": 34,
        }

        for model_name, expected_count in expected_counts.items():
            model = build_model(
                model_name=model_name,
                input_channels=3,
                num_classes=10,
                image_size=32,
            )

            candidates = get_candidate_layer_prefixes(model)

            # ResNet 的层数按主干卷积和最终 fc 统计，残差投影分支不参与选层。
            self.assertEqual(len(candidates), expected_count)
            self.assertFalse(any(".downsample." in layer for layer in candidates))
            self.assertFalse(any(".shortcut." in layer for layer in candidates))

    def test_zero_updates_do_not_produce_fake_cosine_anomaly(self) -> None:
        benign_delta = {
            "fc.weight": torch.zeros(2, 2),
            "fc.bias": torch.zeros(2),
        }
        malicious_delta = {
            "fc.weight": torch.zeros(2, 2),
            "fc.bias": torch.zeros(2),
        }

        metrics = compute_round_layer_metrics(
            candidate_layers=["fc"],
            benign_delta=benign_delta,
            malicious_delta=malicious_delta,
            round_index=1,
            epsilon=1e-12,
        )

        self.assertEqual(metrics.cosine_distances["fc"], 0.0)
        self.assertEqual(metrics.combined_scores["fc"], 0.0)

    def test_adaptive_weight_uses_raw_variance_not_zscore_variance(self) -> None:
        magnitudes = {"l1": 1.0, "l2": 5.0, "l3": 9.0}
        cosine_distances = {"l1": 0.10, "l2": 0.11, "l3": 0.09}

        magnitude_z = zscore_layer_values(magnitudes, epsilon=1e-12)
        cosine_z = zscore_layer_values(cosine_distances, epsilon=1e-12)
        alpha, beta = compute_adaptive_weights(
            magnitudes=magnitudes,
            cosine_distances=cosine_distances,
            epsilon=1e-12,
        )

        self.assertAlmostEqual(
            compute_population_variance(list(magnitude_z.values())),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            compute_population_variance(list(cosine_z.values())),
            1.0,
            places=6,
        )
        self.assertGreater(alpha, beta)
        self.assertGreater(alpha, 0.9)

    def test_selection_prefers_consensus_score_over_top1_truncation(self) -> None:
        summaries = [
            AttackLayerSummary(
                attack_name="attack_a",
                round_metrics=[],
                layer_scores={"conv1": 5.0, "conv2": 4.9, "fc": 4.8},
                top1_layer="conv1",
                top1_score=5.0,
            ),
            AttackLayerSummary(
                attack_name="attack_b",
                round_metrics=[],
                layer_scores={"conv1": -0.2, "conv2": 4.0, "fc": 3.9},
                top1_layer="conv2",
                top1_score=4.0,
            ),
            AttackLayerSummary(
                attack_name="attack_c",
                round_metrics=[],
                layer_scores={"conv1": -0.1, "conv2": -0.2, "fc": 3.8},
                top1_layer="fc",
                top1_score=3.8,
            ),
        ]

        result = select_layers(
            model_name="lenet5",
            dataset_name="mnist",
            partition_name="iid",
            num_rounds=3,
            candidate_layers=["conv1", "conv2", "fc"],
            attack_summaries=summaries,
            k=1,
            weighting_mode="raw_variance_on_zscored_scores",
        )

        # 这里故意构造一个场景：
        # `conv1` 的单次 Top-1 分数最高，但 `fc` 的跨攻击共识分更高，
        # 因此最终应该按共识优先选出 `fc`。
        self.assertEqual(result.selected_layers, ["fc"])
        self.assertIn("attack_a", result.dropped_top1_by_attack)
        self.assertFalse(result.top1_by_attack["attack_a"]["selected_in_final_set"])


if __name__ == "__main__":
    unittest.main()
