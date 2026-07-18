import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ksllm4rec_grpo.trie import DOMAIN_ORDER, SidPrefixTrie
from ksllm4rec_orpo.data import Sid


class SidPrefixTrieTest(unittest.TestCase):
    def setUp(self):
        self.sids = [
            Sid("living", 7, 8, 9),
            Sid("video", 1, 2, 9),
            Sid("ad", 9, 9, 9),
            Sid("video", 1, 8, 7),
            Sid("prod", 4, 5, 6),
            Sid("video", 1, 2, 3),
            Sid("video", 2, 0, 8_191),
            Sid("video", 1, 2, 3),
        ]
        self.trie = SidPrefixTrie.from_sids(self.sids)

    def test_builds_fixed_domain_order_and_deduplicates_leaves(self):
        self.assertEqual(DOMAIN_ORDER, ("video", "prod", "ad", "living"))
        self.assertEqual(self.trie.leaf_count, 7)
        self.assertEqual(self.trie.allowed_a("video").tolist(), [1, 2])
        self.assertEqual(self.trie.allowed_a("prod").tolist(), [4])
        self.assertEqual(self.trie.allowed_a("ad").tolist(), [9])
        self.assertEqual(self.trie.allowed_a("living").tolist(), [7])
        self.assertEqual(self.trie.allowed_b("video", 1).tolist(), [2, 8])
        self.assertEqual(self.trie.allowed_c("video", 1, 2).tolist(), [3, 9])

    def test_membership_and_missing_prefix_queries(self):
        for sid in set(self.sids):
            self.assertTrue(self.trie.contains(sid))
        self.assertFalse(self.trie.contains(Sid("video", 1, 2, 4)))
        self.assertEqual(self.trie.allowed_b("video", 8_190).dtype, np.uint16)
        self.assertEqual(self.trie.allowed_b("video", 8_190).size, 0)
        self.assertEqual(self.trie.allowed_c("video", 1, 7).dtype, np.uint16)
        self.assertEqual(self.trie.allowed_c("video", 1, 7).size, 0)

    def test_array_shapes_dtypes_and_counts(self):
        expected_dtypes = {
            "domain_a_offsets": np.int64,
            "a_values": np.uint16,
            "da_b_offsets": np.int64,
            "b_values": np.uint16,
            "dab_c_offsets": np.int64,
            "c_values": np.uint16,
        }
        for name, dtype in expected_dtypes.items():
            array = getattr(self.trie, name)
            self.assertEqual(array.dtype, dtype)
            self.assertEqual(array.ndim, 1)
            self.assertFalse(array.flags.writeable)
        self.assertEqual(self.trie.domain_a_offsets.shape, (5,))
        self.assertEqual(self.trie.da_b_offsets.shape, (self.trie.a_values.size + 1,))
        self.assertEqual(self.trie.dab_c_offsets.shape, (self.trie.b_values.size + 1,))
        self.assertEqual(self.trie.counts["leaves"], 7)
        self.assertEqual(self.trie.counts["by_domain"]["video"]["leaves"], 4)
        self.trie.validate(expected_leaf_count=7)
        with self.assertRaisesRegex(ValueError, "leaf count mismatch"):
            self.trie.validate(expected_leaf_count=8)

    def test_save_load_round_trip_with_metadata_and_integrity_manifest(self):
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "trie"
            metadata = {"source_sha256": "abc123", "source_rows": 32_705}
            manifest = self.trie.save(output, metadata=metadata)
            loaded = SidPrefixTrie.load(output, expected_leaf_count=7)

            self.assertEqual(manifest["metadata"], metadata)
            self.assertEqual(loaded.metadata, metadata)
            self.assertEqual(loaded.counts, self.trie.counts)
            for sid in set(self.sids):
                self.assertTrue(loaded.contains(sid))
            for name, entry in manifest["arrays"].items():
                self.assertEqual(entry["shape"], list(getattr(self.trie, name).shape))
                self.assertEqual(entry["dtype"], str(getattr(self.trie, name).dtype))
                self.assertEqual(len(entry["sha256"]), 64)

    def test_load_rejects_manifest_leaf_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "trie"
            self.trie.save(output)
            path = output / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["counts"]["leaves"] += 1
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "counts"):
                SidPrefixTrie.load(output)

    def test_load_rejects_modified_array(self):
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "trie"
            self.trie.save(output)
            path = output / "c_values.npy"
            values = np.load(path, allow_pickle=False)
            values[0] = 42
            np.save(path, values, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "integrity"):
                SidPrefixTrie.load(output)

    def test_constructor_rejects_wrong_dtype_and_unsorted_siblings(self):
        arrays = {name: array.copy() for name, array in self.trie.arrays.items()}
        arrays["c_values"] = arrays["c_values"].astype(np.int64)
        with self.assertRaisesRegex(ValueError, "uint16"):
            SidPrefixTrie(**arrays)

        arrays = {name: array.copy() for name, array in self.trie.arrays.items()}
        first_group_start = int(arrays["dab_c_offsets"][0])
        arrays["c_values"][first_group_start : first_group_start + 2] = [9, 3]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            SidPrefixTrie(**arrays)

    def test_rejects_empty_invalid_domain_and_out_of_range_components(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            SidPrefixTrie.from_sids([])
        with self.assertRaisesRegex(ValueError, "domain"):
            SidPrefixTrie.from_sids([Sid("unknown", 1, 2, 3)])
        with self.assertRaisesRegex(ValueError, "within"):
            SidPrefixTrie.from_sids([Sid("video", 8_192, 2, 3)])
        with self.assertRaisesRegex(TypeError, "integer"):
            SidPrefixTrie.from_sids([Sid("video", True, 2, 3)])


if __name__ == "__main__":
    unittest.main()
