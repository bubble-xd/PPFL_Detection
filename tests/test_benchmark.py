from __future__ import annotations

import unittest

from utils.benchmark import (
    _resolve_fedavg_feature_mode,
    _resolve_robust_method_feature_runs,
    _summarize_round_logs,
)


class BenchmarkSchedulingTestCase(unittest.TestCase):
    def test_resolve_fedavg_feature_mode_prefers_raw_full(self) -> None:
        feature_mode = _resolve_fedavg_feature_mode(
            ["selected_layers", "raw_full", "selected_layers_projected"]
        )

        self.assertEqual(feature_mode, "raw_full")

    def test_resolve_robust_method_feature_runs_collapses_fedavg(self) -> None:
        runs = _resolve_robust_method_feature_runs(
            methods=["fedavg", "krum"],
            feature_modes=[
                "raw_full",
                "selected_layers",
                "selected_layers_balanced",
                "selected_layers_projected",
                "selected_layers_balanced_projected",
            ],
        )

        # `fedavg` 不依赖特征，因此这里应只保留一组调度；
        # `krum` 仍然需要完整扫描全部特征模式。
        self.assertEqual(
            runs,
            [
                ("fedavg", "raw_full"),
                ("krum", "raw_full"),
                ("krum", "selected_layers"),
                ("krum", "selected_layers_balanced"),
                ("krum", "selected_layers_projected"),
                ("krum", "selected_layers_balanced_projected"),
            ],
        )

    def test_resolve_robust_method_feature_runs_falls_back_to_first_feature(self) -> None:
        runs = _resolve_robust_method_feature_runs(
            methods=["fedavg"],
            feature_modes=["selected_layers", "selected_layers_projected"],
        )

        self.assertEqual(runs, [("fedavg", "selected_layers")])

    def test_summarize_round_logs_uses_tail_mean_asr(self) -> None:
        round_logs = [
            {"f1": 0.1, "acc": 0.7, "asr": 0.2, "bm_gap": 0.3},
            {"f1": 0.2, "acc": 0.8, "asr": 0.6, "bm_gap": 0.4},
            {"f1": 0.3, "acc": 0.9, "asr": 0.8, "bm_gap": 0.5},
        ]

        summary = _summarize_round_logs(round_logs, asr_tail_rounds=2)

        # ASR 汇总取最后 N 轮均值，避免 Excel 只受最后一轮偶然值支配。
        self.assertAlmostEqual(summary["final_asr"], 0.7, places=6)
        self.assertAlmostEqual(summary["tail_mean_asr"], 0.7, places=6)
        self.assertAlmostEqual(summary["last_asr"], 0.8, places=6)
        self.assertAlmostEqual(summary["final_acc"], 0.9, places=6)


if __name__ == "__main__":
    unittest.main()
