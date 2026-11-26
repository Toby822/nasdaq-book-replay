import tempfile
import unittest
from pathlib import Path

import numpy as np

from lob.replay import FIELDS, Replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.times = np.array([10, 11, 11, 11, 13], dtype=np.int64)
        self.book = np.zeros((len(self.times), FIELDS), dtype=np.int32)
        self.book[:, 0] = [50, 60, 40, 75, 100]
        self.book[:, 13] = [2, 4, 4, 2, 0]
        self.replay = Replay.encode(self.times, self.book, stride=2)

    def test_as_of_preserves_feed_order_at_a_checkpoint_tie(self):
        np.testing.assert_array_equal(self.replay.as_of(11), self.book[3])
        np.testing.assert_array_equal(self.replay.as_of(11, before=True), self.book[0])
        np.testing.assert_array_equal(self.replay.as_of(12), self.book[3])
        self.assertIsNone(self.replay.as_of(9))

    def test_replay_deltas_can_increase_and_decrease_sizes(self):
        for start, stop in [(0, 2), (2, 4), (4, 5)]:
            np.testing.assert_array_equal(self.replay.decode(start, stop), self.book[start:stop])
        with self.assertRaises(ValueError):
            self.replay.decode(1, 3)

    def test_future_snapshot_change_does_not_affect_prior_lookup(self):
        changed = self.book.copy()
        changed[4, 0] = 999
        second = Replay.encode(self.times, changed, stride=2)
        np.testing.assert_array_equal(second.as_of(11), self.replay.as_of(11))

    def test_save_load_preserves_replay(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "test.npz"
            self.replay.save(path)
            second = Replay.load(path)
            self.assertEqual(second.stride, 2)
            np.testing.assert_array_equal(second.as_of(13), self.book[4])

    def test_stream_handles_a_range_across_checkpoints(self):
        np.testing.assert_array_equal(list(self.replay.iter_rows(1, 5)), self.book[1:5])
        self.assertEqual(list(self.replay.iter_rows(2, 2)), [])
        with self.assertRaises(IndexError):
            list(self.replay.iter_rows(0, 6))

    def test_rejects_out_of_order_times(self):
        with self.assertRaises(ValueError):
            Replay.encode(self.times[::-1], self.book)


if __name__ == "__main__":
    unittest.main()
