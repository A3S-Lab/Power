from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


SCRIPT_PATH = Path(__file__).with_name("add-gguf-mtp-head.py")
SPEC = importlib.util.spec_from_file_location("add_gguf_mtp_head", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FrequencyTokenTests(unittest.TestCase):
    def test_loads_metadata_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frequency.json"
            path.write_text(json.dumps({"token_ids": [4, 2, 7]}), encoding="utf-8")

            self.assertEqual(MODULE.load_frequency_token_ids(path), [4, 2, 7])

    def test_applies_limit_and_preserves_rank_order(self) -> None:
        token_ids = MODULE.validate_frequency_token_ids([4, 2, 7], 8, 2)

        np.testing.assert_array_equal(token_ids, np.asarray([4, 2], dtype=np.int64))

    def test_rejects_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            MODULE.validate_frequency_token_ids([4, 2, 4], 8, None)

    def test_rejects_out_of_range_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            MODULE.validate_frequency_token_ids([4, 8], 8, None)

    def test_selects_complete_quantized_rows(self) -> None:
        rows = np.arange(24, dtype=np.uint8).reshape(6, 4)
        token_ids = np.asarray([4, 1, 5], dtype=np.int64)

        selected = MODULE.select_frequency_ranked_rows(rows, token_ids, 6)

        np.testing.assert_array_equal(selected, rows[[4, 1, 5]])
        self.assertTrue(selected.flags.c_contiguous)

    def test_replace_filters_old_head_and_mapping(self) -> None:
        tensors = [
            SimpleNamespace(name="output.weight"),
            SimpleNamespace(name=MODULE.MTP_HEAD_NAME),
            SimpleNamespace(name=MODULE.DRAFT_TO_TARGET_NAME),
        ]

        selected = MODULE.tensors_to_copy(tensors, replace_existing_head=True)

        self.assertEqual([tensor.name for tensor in selected], ["output.weight"])

    def test_fr_metadata_keys_are_replaced_as_one_identity(self) -> None:
        self.assertEqual(
            MODULE.FR_METADATA_KEYS,
            {
                "a3s.mtp.fr.mapping",
                "a3s.mtp.fr.vocabulary_size",
                "a3s.mtp.fr.d2t_sha256",
            },
        )


if __name__ == "__main__":
    unittest.main()
