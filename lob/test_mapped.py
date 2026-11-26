"""Small checks for the persistent format, including damaged archives."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lob.mapped import open_mapped, save_mapped
from lob.replay import FIELDS, Replay


class MappedTests(unittest.TestCase):
    def test_round_trip_reads_a_checkpoint_boundary(self):
        depth = np.zeros((5, FIELDS), dtype=np.int32)
        depth[:, 0] = [32, 28, 30, 27, 24]
        archive = Replay.encode(np.array([1, 2, 2, 2, 3]), depth, stride=2)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "sample.lob"
            save_mapped(archive, path)
            mapped = open_mapped(path)
            self.assertIsInstance(mapped.times, np.memmap)
            self.assertFalse(mapped.times.flags.writeable)
            np.testing.assert_array_equal(mapped.as_of(2), depth[3])
            np.testing.assert_array_equal(mapped.as_of(2, before=True), depth[0])
            np.testing.assert_array_equal(list(mapped.iter_rows(1, 5)), depth[1:5])

    def test_rejects_truncated_archive(self):
        depth = np.zeros((3, FIELDS), dtype=np.int32)
        depth[:, 0] = [40, 38, 41]
        archive = Replay.encode(np.array([1, 2, 3]), depth, stride=2)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "sample.lob"
            save_mapped(archive, path)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "Truncated"):
                open_mapped(path)

    def test_all_zero_delta_archive(self):
        depth = np.ones((4, FIELDS), dtype=np.int32)
        archive = Replay.encode(np.arange(4, dtype=np.int64), depth, stride=3)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "sample.lob"
            save_mapped(archive, path)
            mapped = open_mapped(path)
            np.testing.assert_array_equal(list(mapped.iter_rows(0, 4)), depth)
