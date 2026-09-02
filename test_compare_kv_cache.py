import io
import unittest
from collections import Counter

from compare_kv_cache import (
    build_rows,
    estimate_v32,
    estimate_v4_flash,
    load_model_configs,
    parse_token_count,
    print_csv,
)


class CacheEstimateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v32, cls.v4 = load_model_configs()

    def test_configs_match_released_architectures(self):
        self.assertEqual(self.v32["num_hidden_layers"], 61)
        self.assertEqual(self.v4["num_hidden_layers"], 43)
        ratios = Counter(self.v4["compress_ratios"][:43])
        self.assertEqual(ratios, Counter({4: 21, 128: 20, 0: 2}))

    def test_one_mi_token_paper_result(self):
        tokens = 1 << 20
        v32 = estimate_v32(tokens, self.v32)
        v4 = estimate_v4_flash(tokens, self.v4)
        self.assertEqual(v32.main_kv_bytes, 41_959_817_216)
        self.assertEqual(v32.indexer_k_bytes, 8_443_133_952)
        self.assertEqual(v32.total_bytes, 50_402_951_168)
        self.assertEqual(v4.main_kv_bytes, 3_310_616_576)
        self.assertEqual(v4.indexer_k_bytes, 374_341_632)
        self.assertEqual(v4.sliding_window_bytes, 3_214_336)
        self.assertEqual(v4.total_bytes, 3_688_172_544)
        self.assertAlmostEqual(v4.total_bytes / v32.total_bytes, 0.07317, places=5)
        self.assertAlmostEqual(v32.total_bytes / v4.total_bytes, 13.666, places=3)

    def test_v4_keeps_only_complete_compression_blocks(self):
        before = estimate_v4_flash(127, self.v4)
        after = estimate_v4_flash(128, self.v4)
        expected_new_bytes = 43 * 584 + 21 * 584 + 21 * 68 + 20 * 584
        self.assertEqual(after.total_bytes - before.total_bytes, expected_new_bytes)

    def test_cluster_bytes_are_replica_sum(self):
        row = build_rows([4096], gpus=8)[0]
        self.assertEqual(row["v32_bytes_cluster"], row["v32_bytes_per_gpu"] * 8)
        self.assertEqual(
            row["v4_flash_bytes_cluster"], row["v4_flash_bytes_per_gpu"] * 8
        )

    def test_token_suffixes(self):
        self.assertEqual(parse_token_count("1Mi"), 1 << 20)
        self.assertEqual(parse_token_count("256K"), 256_000)
        self.assertEqual(parse_token_count("1_000_000"), 1_000_000)

    def test_csv(self):
        output = io.StringIO()
        print_csv(build_rows([4096]), output)
        self.assertIn("v4_over_v32_percent", output.getvalue())


if __name__ == "__main__":
    unittest.main()
