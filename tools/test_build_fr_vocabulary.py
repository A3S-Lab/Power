from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).with_name("build-fr-vocabulary.py")
SPEC = importlib.util.spec_from_file_location("build_fr_vocabulary", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FrequencyRankingTests(unittest.TestCase):
    def test_ranks_by_frequency_then_token_id(self) -> None:
        counts = Counter({7: 2, 4: 5, 2: 2, 1: 1})

        selected = MODULE.select_ranked_token_ids(counts, [], 8, 3)

        self.assertEqual(selected, [4, 2, 7])

    def test_reserves_missing_special_tokens(self) -> None:
        counts = Counter({1: 10, 2: 9, 3: 8, 4: 7, 5: 6})

        selected = MODULE.select_ranked_token_ids(counts, [5], 8, 3)

        self.assertEqual(selected, [1, 2, 5])

    def test_rejects_insufficient_corpus_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique tokens"):
            MODULE.select_ranked_token_ids(Counter({1: 2, 2: 1}), [], 8, 3)


if __name__ == "__main__":
    unittest.main()
