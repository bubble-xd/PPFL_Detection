from __future__ import annotations

import unittest

from attacks.targeted.badnets import BadNetsAttack
from attacks.targeted.dba import DBAAttack
from config import Config
from data.data_loader import _get_dataset_info
from models import build_model


class Cifar100SupportTestCase(unittest.TestCase):
    def test_cifar100_dataset_info_matches_resnet34_requirements(self) -> None:
        info = _get_dataset_info("cifar100")

        self.assertEqual(info["num_classes"], 100)
        self.assertEqual(info["input_channels"], 3)
        self.assertEqual(info["image_size"], 32)

        model = build_model(
            model_name="resnet34",
            input_channels=info["input_channels"],
            num_classes=info["num_classes"],
            image_size=info["image_size"],
        )

        self.assertEqual(model.fc.out_features, 100)

    def test_cifar100_display_and_attack_overrides_are_configured(self) -> None:
        self.assertEqual(Config.DATASET_DISPLAY_NAMES["cifar100"], "CIFAR100")
        self.assertEqual(
            Config.ATTACK_STRENGTHS_BY_DATASET["edge_case"]["cifar100"]["target_label"],
            99,
        )
        self.assertEqual(
            Config.ATTACK_STRENGTHS_BY_DATASET["semantic_backdoor"]["cifar100"]["semantic_source"],
            "southwest",
        )

    def test_cifar100_trigger_attacks_use_100_classes(self) -> None:
        badnets = BadNetsAttack(
            dataset_name="cifar100",
            target_label=99,
            poisoning_ratio=0.1,
            trigger_size=3,
        )
        dba = DBAAttack(
            dataset_name="cifar100",
            target_label=99,
            poisoning_ratio=0.1,
            trigger_size=3,
        )

        self.assertEqual(badnets.num_classes, 100)
        self.assertEqual(dba.num_classes, 100)
        self.assertEqual(tuple(badnets.trigger.shape), (3, 3, 3))
        self.assertEqual(tuple(dba.trigger.shape), (3, 3, 3))


if __name__ == "__main__":
    unittest.main()
