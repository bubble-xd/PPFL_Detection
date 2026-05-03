from __future__ import annotations

from pathlib import Path
import unittest

import torch
from torch.utils.data import TensorDataset

from attacks.adapters import LabelFlippingTargetedAdapter
from attacks.registry import build_attack_adapter
from attacks.targeted.badnets import BadNetsAttack
from attacks.targeted.dba import DBAAttack
from attacks.targeted.edge_case import ARDISDataset, EDGE_CASE_DATA_DIR
from attacks.targeted.semantic_backdoor import SemanticBackdoorAttack


def _has_ardis_files() -> bool:
    root = Path(EDGE_CASE_DATA_DIR)
    return all(
        (root / filename).exists()
        for filename in (
            "ARDIS_train_2828.csv",
            "ARDIS_train_labels.csv",
            "ARDIS_test_2828.csv",
            "ARDIS_test_labels.csv",
        )
    )


class TriggerAsrEvalTestCase(unittest.TestCase):
    def _build_mnist_dataset(self) -> TensorDataset:
        images = torch.zeros(5, 1, 28, 28)
        labels = torch.tensor([1, 0, 1, 2, 3], dtype=torch.long)
        return TensorDataset(images, labels)

    def _assert_asr_dataset_filters_clean_target_class(self, attack) -> None:
        poisoned = attack.poison_dataset(self._build_mnist_dataset(), train=False)
        images, labels = poisoned.tensors

        # 评估集应只包含原始非目标类样本，避免干净 target 类样本被 ASR 当作成功攻击。
        self.assertEqual(len(labels), 3)
        self.assertTrue(torch.equal(labels, torch.full_like(labels, attack.target_label)))

        trigger = attack.trigger.to(dtype=images.dtype).unsqueeze(0)
        trigger_region = images[:, :, -attack.trigger_size :, -attack.trigger_size :]
        self.assertTrue(torch.allclose(trigger_region, trigger.expand_as(trigger_region)))

    def test_badnets_asr_eval_excludes_original_target_class(self) -> None:
        attack = BadNetsAttack(
            dataset_name="mnist",
            target_label=1,
            poisoning_ratio=1.0,
            trigger_size=2,
        )

        self._assert_asr_dataset_filters_clean_target_class(attack)

    def test_dba_asr_eval_excludes_original_target_class(self) -> None:
        attack = DBAAttack(
            dataset_name="mnist",
            target_label=1,
            poisoning_ratio=1.0,
            trigger_size=2,
            shard_id=0,
            num_shards=2,
        )

        self._assert_asr_dataset_filters_clean_target_class(attack)

    def test_label_flipping_targeted_asr_eval_uses_source_class_only(self) -> None:
        images = torch.arange(4 * 1 * 2 * 2, dtype=torch.float32).reshape(4, 1, 2, 2)
        labels = torch.tensor([4, 1, 4, 9], dtype=torch.long)
        adapter = LabelFlippingTargetedAdapter(
            {"source_class": 4, "target_class": 9, "poison_ratio": 1.0}
        )

        loader = adapter.build_asr_eval_loader(TensorDataset(images, labels), batch_size=2)
        asr_images, asr_labels = loader.dataset.tensors

        # Label flipping 没有触发器，ASR 应评估 source 类被预测成 target 类的比例。
        self.assertTrue(torch.equal(asr_images, images[labels == 4]))
        self.assertTrue(torch.equal(asr_labels, torch.full_like(asr_labels, 9)))

    def test_label_flipping_targeted_rejects_classes_outside_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_class=42"):
            LabelFlippingTargetedAdapter(
                {"source_class": 42, "target_class": 88, "poison_ratio": 1.0},
                num_classes=10,
            )

    def test_label_flipping_targeted_uses_dataset_override(self) -> None:
        adapter = build_attack_adapter(
            attack_config={
                "name": "label_flipping_targeted",
                "params": {"source_class": 42, "target_class": 88, "poison_ratio": 1.0},
                "params_by_dataset": {
                    "cifar10": {"source_class": 3, "target_class": 5},
                },
            },
            dataset_name="cifar10",
            dataset_info={"num_classes": 10},
        )

        # CIFAR10 运行时应覆盖掉 CIFAR100 的 42->88，避免攻击配置静默失效。
        self.assertEqual(adapter.source_class, 3)
        self.assertEqual(adapter.target_class, 5)

    @unittest.skipUnless(_has_ardis_files(), "ARDIS edge-case data files are missing.")
    def test_edge_case_ardis_asr_eval_uses_held_out_source_class(self) -> None:
        dataset = ARDISDataset(target_label=1)
        expected_mask = dataset.test_labels == dataset.source_label
        poisoned = dataset.get_poisoned_testset()
        images, labels = poisoned.tensors

        # Edge-case 训练使用 ARDIS 源类别 7，ASR 也应只使用 held-out 的同源类别样本。
        self.assertEqual(len(images), int(expected_mask.sum().item()))
        self.assertGreater(len(images), 0)
        self.assertTrue(torch.equal(images, dataset.test_images[expected_mask]))
        self.assertTrue(torch.equal(labels, torch.full_like(labels, dataset.target_label)))

    @unittest.skipUnless(_has_ardis_files(), "ARDIS edge-case data files are missing.")
    def test_semantic_backdoor_asr_eval_uses_held_out_semantic_samples(self) -> None:
        attack = SemanticBackdoorAttack(
            dataset_name="mnist",
            target_label=1,
            poisoning_ratio=0.3,
            semantic_source="ardis",
        )
        reference = TensorDataset(
            torch.zeros(2, 1, 28, 28),
            torch.zeros(2, dtype=torch.long),
        )

        eval_dataset = attack.poison_dataset(reference, train=False)
        images, labels = eval_dataset.tensors

        # 语义后门 ASR 不能复用训练注入池，否则会把记忆训练样本当成攻击泛化能力。
        self.assertEqual(len(attack.semantic_samples), len(attack.train_semantic_samples))
        self.assertEqual(len(images), len(attack.eval_semantic_samples))
        self.assertTrue(torch.equal(images, attack.eval_semantic_samples))
        self.assertTrue(torch.equal(labels, torch.full_like(labels, attack.target_label)))


if __name__ == "__main__":
    unittest.main()
