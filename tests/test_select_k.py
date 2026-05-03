from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from layer_extraction.select_k.elbow import (
    find_score_paths,
    find_selection_paths,
    select_k_from_plot_data_file,
    select_k_from_selection_file,
    select_k_from_selection_payload,
)


class SelectKTestCase(unittest.TestCase):
    def _payload(self):
        return {
            "model": "toy",
            "dataset": "toyset",
            "partition": "iid",
            "k": 2,
            "candidate_layers": ["a", "b", "c", "d", "e"],
            "selected_layers": ["a", "b"],
            "consensus_scores": {
                "a": 10.0,
                "b": 9.0,
                "c": 8.0,
                "d": 3.0,
                "e": 2.8,
            },
        }

    def test_chord_elbow_uses_farthest_rank_from_score_line(self) -> None:
        result = select_k_from_selection_payload(self._payload(), method="chord")

        # 最大曲率点出现在 c 和 d 的断层之后，因此自动 k 应为 3。
        self.assertEqual(result.recommended_k, 3)
        self.assertEqual(result.selected_layers, ["a", "b", "c"])

    def test_max_gap_elbow_uses_largest_adjacent_drop(self) -> None:
        result = select_k_from_selection_payload(self._payload(), method="max_gap")

        # max_gap 与 chord 在该构造曲线上都应落在第三层之后。
        self.assertEqual(result.recommended_k, 3)
        self.assertEqual(result.selected_layers, ["a", "b", "c"])

    def test_min_and_max_k_bound_auto_search(self) -> None:
        payload = self._payload()
        payload["consensus_scores"] = {
            "a": 10.0,
            "b": 4.0,
            "c": 3.9,
            "d": 3.8,
            "e": 3.7,
        }

        result = select_k_from_selection_payload(
            payload,
            method="max_gap",
            min_k=2,
            max_k=4,
        )

        # min_k 用于避免第一层过强时退化成只选一个层。
        self.assertEqual(result.recommended_k, 2)

    def test_reads_existing_selection_json_without_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "layer_extraction_foo"
            run_dir.mkdir()
            selection_path = run_dir / "selection.json"
            selection_path.write_text(json.dumps(self._payload()), encoding="utf-8")

            paths = find_selection_paths(tmp_dir)
            result = select_k_from_selection_file(paths[0])

        self.assertEqual(paths, [selection_path])
        self.assertEqual(result.model, "toy")
        self.assertEqual(result.recommended_k, 3)

    def test_reads_plot_data_csv_by_deduplicating_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "toy_plot_data.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "dataset,model_name,attack_type,layer,consensus_score,consensus_rank,is_selected",
                        "toyset,toy,attack_a,a,10.0,1,1",
                        "toyset,toy,attack_a,b,9.0,2,1",
                        "toyset,toy,attack_a,c,8.0,3,0",
                        "toyset,toy,attack_a,d,3.0,4,0",
                        "toyset,toy,attack_a,e,2.8,5,0",
                        "toyset,toy,attack_b,a,10.0,1,1",
                    ]
                ),
                encoding="utf-8",
            )

            paths = find_score_paths(tmp_dir)
            result = select_k_from_plot_data_file(paths[0], method="chord")

        self.assertEqual(paths, [csv_path])
        self.assertEqual(result.existing_k, 2)
        self.assertEqual(result.recommended_k, 3)
        self.assertEqual(result.selected_layers, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
