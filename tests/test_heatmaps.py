from __future__ import annotations

import csv
import tempfile
import unittest
from unittest import mock

import torch

from utils.heatmaps import _resolve_heatmap_grid, save_cosine_heatmaps


class HeatmapExportTestCase(unittest.TestCase):
    def test_resolve_heatmap_grid_expands_for_five_feature_modes(self) -> None:
        self.assertEqual(_resolve_heatmap_grid(5), (2, 3))

    def test_save_cosine_heatmaps_persists_plot_and_editable_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_paths = save_cosine_heatmaps(
                output_dir=temp_dir,
                experiment_name="ToyExperiment",
                attack_name="badnets",
                attack_mode="targeted",
                round_idx=3,
                feature_matrices={
                    "raw_full": torch.tensor(
                        [
                            [1.0, 0.0],
                            [0.9, 0.1],
                            [-1.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                    "selected_layers": torch.tensor(
                        [
                            [1.0, 0.0],
                            [0.95, 0.05],
                            [-0.8, -0.2],
                        ],
                        dtype=torch.float32,
                    ),
                },
                client_order=[0, 1, 2],
                malicious_ids=[2],
                feature_display_names={
                    "raw_full": "原始",
                    "selected_layers": "提取",
                },
                artifact_subdir="shared_global_base",
                similarity_space_tag="shared_global_base_local_models",
                similarity_space_description="Cosine similarity on local client models with shared global base",
            )

            for path in saved_paths:
                self.assertTrue(path)
                self.assertTrue(path.endswith((".png", ".csv")))

            data_csv_path = next(path for path in saved_paths if path.endswith("_heatmap_data.csv"))

            with open(data_csv_path, "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 18)
            self.assertEqual({row["feature_mode"] for row in rows}, {"raw_full", "selected_layers"})
            self.assertEqual({row["similarity_space_tag"] for row in rows}, {"shared_global_base_local_models"})
            self.assertEqual({row["row_is_malicious"] for row in rows if row["row_client_id"] == "2"}, {"1"})
            self.assertEqual({row["col_is_malicious"] for row in rows if row["col_client_id"] == "2"}, {"1"})
            self.assertTrue(all(row["bm_gap"] for row in rows))
            self.assertTrue(all(row["cosine_similarity"] for row in rows))

    def test_save_cosine_heatmaps_plots_all_feature_modes(self) -> None:
        feature_matrices = {
            "raw_full": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
            "selected_layers": torch.tensor([[1.0, 0.0], [0.2, 0.8], [0.8, 0.2]], dtype=torch.float32),
            "selected_layers_balanced": torch.tensor([[0.9, 0.1], [0.1, 0.9], [1.0, 0.0]], dtype=torch.float32),
            "selected_layers_projected": torch.tensor([[0.6, 0.4], [0.4, 0.6], [0.9, 0.1]], dtype=torch.float32),
            "selected_layers_balanced_projected": torch.tensor(
                [[0.7, 0.3], [0.3, 0.7], [0.2, 0.8]],
                dtype=torch.float32,
            ),
        }
        feature_display_names = {feature_mode: feature_mode for feature_mode in feature_matrices.keys()}

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch(
                "matplotlib.axes._axes.Axes.set_title",
                autospec=True,
            ) as mocked_set_title:
                save_cosine_heatmaps(
                    output_dir=temp_dir,
                    experiment_name="ToyExperiment",
                    attack_name="badnets",
                    attack_mode="targeted",
                    round_idx=1,
                    feature_matrices=feature_matrices,
                    client_order=[0, 1, 2],
                    malicious_ids=[2],
                    feature_display_names=feature_display_names,
                )

        self.assertEqual(mocked_set_title.call_count, len(feature_matrices))
        self.assertTrue(
            all("BM-Gap" not in call.args[1] for call in mocked_set_title.call_args_list)
        )

    def test_save_cosine_heatmaps_respects_client_order_in_exported_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_paths = save_cosine_heatmaps(
                output_dir=temp_dir,
                experiment_name="ToyExperiment",
                attack_name="badnets",
                attack_mode="targeted",
                round_idx=2,
                feature_matrices={
                    "raw_full": torch.tensor(
                        [
                            [1.0, 0.0],
                            [0.0, 1.0],
                            [1.0, 1.0],
                        ],
                        dtype=torch.float32,
                    ),
                },
                client_order=[2, 0, 1],
                malicious_ids=[1],
                feature_display_names={"raw_full": "原始"},
            )

            data_csv_path = next(path for path in saved_paths if path.endswith("_heatmap_data.csv"))
            with open(data_csv_path, "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        target_row = next(
            row
            for row in rows
            if row["feature_mode"] == "raw_full"
            and row["row_position"] == "0"
            and row["col_position"] == "1"
        )

        # client_order=[2, 0, 1] 时，导出矩阵的 (0, 1) 应对应客户端 2 与 0 的余弦相似度。
        self.assertEqual(target_row["row_client_id"], "2")
        self.assertEqual(target_row["col_client_id"], "0")
        self.assertAlmostEqual(float(target_row["cosine_similarity"]), 0.707107, places=5)


if __name__ == "__main__":
    unittest.main()
