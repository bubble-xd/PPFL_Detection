from __future__ import annotations

import unittest

from config import Config


class ConfigPoisonRateTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.original_poison_rate = Config.POISON_RATE
        self.original_num_clients = Config.NUM_CLIENTS
        self.original_multi_krum_m = Config.MULTI_KRUM_M

    def tearDown(self) -> None:
        Config.POISON_RATE = self.original_poison_rate
        Config.NUM_CLIENTS = self.original_num_clients
        Config.MULTI_KRUM_M = self.original_multi_krum_m

    def test_get_poison_rates_accepts_single_float(self) -> None:
        Config.POISON_RATE = 0.2

        self.assertEqual(Config.get_poison_rates(), [0.2])

    def test_get_poison_rates_accepts_list(self) -> None:
        Config.POISON_RATE = [0.1, 0.2, 0.3]

        self.assertEqual(Config.get_poison_rates(), [0.1, 0.2, 0.3])

    def test_get_num_malicious_and_multi_krum_m_support_list_config(self) -> None:
        Config.NUM_CLIENTS = 10
        Config.POISON_RATE = [0.1, 0.3]
        Config.MULTI_KRUM_M = None

        # 列表模式下，旧接口默认按第一项工作；显式传参时使用当前轮的投毒比例。
        self.assertEqual(Config.get_num_malicious(), 1)
        self.assertEqual(Config.get_num_malicious(poison_rate=0.3), 3)
        self.assertEqual(Config.get_multi_krum_m(default_num_malicious=3), 5)


if __name__ == "__main__":
    unittest.main()
