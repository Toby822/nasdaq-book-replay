"""Lossless checkpoint-and-delta replay for official LOBSTER depth snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

FIELDS = 40
CHECKPOINT = 2_048
SYMBOLS = ("AAPL", "AMZN", "MSFT")


def read_sample(archive: Path):
    with ZipFile(archive) as files:
        messages, = [name for name in files.namelist() if "_message_" in name]
        orderbook, = [name for name in files.namelist() if "_orderbook_" in name]
        with files.open(messages) as stream:
            events = pd.read_csv(stream, header=None, usecols=[0, 1], dtype={0: str, 1: np.int8})
        with files.open(orderbook) as stream:
            snapshots = pd.read_csv(stream, header=None, dtype=np.int32).to_numpy(copy=True)
    digits = events[0].str.split(".", n=1, regex=False, expand=True).fillna("")
    fractions = digits[1]
    if fractions.str.len().max() > 12:
        raise ValueError(f"More than 12 fractional timestamp digits in {archive}")
    # Five sample rows have 12 decimal digits. Preserve their exact order.
    picoseconds = digits[0].astype("int64") * 10**12 + fractions.str.ljust(12, "0").astype("int64")
    times = picoseconds.to_numpy()
    if (len(times) != len(snapshots) or snapshots.shape[1] != FIELDS
            or np.any(times[1:] < times[:-1])):
        raise ValueError(f"Unaligned LOBSTER files in {archive}")
    return times, events[1].to_numpy(), snapshots


@dataclass
class Replay:
    times: np.ndarray
    pointers: np.ndarray
    columns: np.ndarray
    deltas: np.ndarray
    checkpoints: np.ndarray
    stride: int

    @classmethod
    def encode(cls, times, snapshots, stride=CHECKPOINT):
        if (len(times) == 0 or len(times) != len(snapshots) or stride < 1
                or snapshots.ndim != 2 or snapshots.shape[1] != FIELDS
                or snapshots.dtype != np.int32 or np.any(times[1:] < times[:-1])):
            raise ValueError("Expected sorted timestamps and aligned int32 snapshots")
        differences = np.diff(snapshots, axis=0, prepend=snapshots[:1]).astype(np.int32)
        # Checkpoints are complete states. Their rows need no delta.
        differences[::stride] = 0
        rows, columns = np.nonzero(differences)
        counts = np.bincount(rows, minlength=len(times))
        pointers = np.r_[0, counts.cumsum()].astype(np.uint32)
        return cls(times, pointers, columns.astype(np.uint8),
                   differences[rows, columns], snapshots[::stride].copy(), stride)

    def decode(self, start, stop):
        """Reconstruct rows [start, stop) in one checkpoint interval."""
        if not (0 <= start < stop <= len(self.times)):
            raise IndexError((start, stop))
        block = start // self.stride
        base = block * self.stride
        if (stop - 1) // self.stride != block:
            raise ValueError("Decode must stay inside one checkpoint interval")
        lo, hi = self.pointers[[base, stop]]
        widths = np.diff(self.pointers[base:stop + 1])
        rows = np.repeat(np.arange(stop - base), widths.astype(np.int64))
        differences = np.zeros((stop - base, FIELDS), dtype=np.int32)
        differences[rows, self.columns[lo:hi]] = self.deltas[lo:hi]
        result = np.cumsum(differences, axis=0, dtype=np.int32)
        result += self.checkpoints[block]
        return result[start - base:]

    def iter_rows(self, start, stop):
        """Yield at most one checkpoint block at a time."""
        if not 0 <= start <= stop <= len(self.times):
            raise IndexError((start, stop))
        while start < stop:
            end = min(stop, (start // self.stride + 1) * self.stride)
            yield from self.decode(start, end)
            start = end

    def as_of(self, timestamp_ps, *, before=False):
        """Default is the last row at or before t; before=True excludes all rows at t."""
        side = "left" if before else "right"
        index = int(self.times.searchsorted(timestamp_ps, side=side)) - 1
        if index < 0:
            return None
        return self.decode(index, index + 1)[0]

    def save(self, path):
        np.savez(path, times=self.times, pointers=self.pointers, columns=self.columns,
                 deltas=self.deltas, checkpoints=self.checkpoints, stride=self.stride)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["times"], data["pointers"], data["columns"], data["deltas"],
                       data["checkpoints"], int(data["stride"]))

    def bytes(self):
        return sum(array.nbytes for array in
                   (self.times, self.pointers, self.columns, self.deltas, self.checkpoints))


def verify(replay, reference, event_types):
    digest = hashlib.sha256()
    for start in range(0, len(reference), replay.stride):
        stop = min(len(reference), start + replay.stride)
        block = replay.decode(start, stop)
        np.testing.assert_array_equal(block, reference[start:stop])
        digest.update(block.astype("<i4", copy=False).tobytes())

    tied = np.flatnonzero(replay.times[1:] == replay.times[:-1])
    assert len(tied) > 0
    indices = np.r_[0, replay.stride - 1, replay.stride, len(reference) - 1,
                    tied[np.linspace(0, len(tied) - 1, 250, dtype=int)]]
    for t in replay.times[indices]:
        last = replay.times.searchsorted(t, side="right") - 1
        first = replay.times.searchsorted(t, side="left")
        np.testing.assert_array_equal(replay.as_of(t), reference[last])
        if first:
            np.testing.assert_array_equal(replay.as_of(t, before=True), reference[first - 1])
        else:
            assert replay.as_of(t, before=True) is None

    # The official feed describes hidden executions as leaving visible depth unchanged.
    hidden = np.flatnonzero(event_types == 5)
    hidden = hidden[hidden > 0]
    np.testing.assert_array_equal(reference[hidden], reference[hidden - 1])
    return digest.hexdigest(), len(tied)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--save-books", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol in SYMBOLS:
        source = args.data / f"{symbol}.zip"
        times, events, book = read_sample(source)
        replay = Replay.encode(times, book)
        digest, ties = verify(replay, book, events)
        if args.save_books:
            path = args.output / f"{symbol}_replay.npz"
            replay.save(path)
            persisted = Replay.load(path)
            assert verify(persisted, book, events) == (digest, ties)
        with source.open("rb") as stream:
            source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append(dict(symbol=symbol, updates=len(book), equal_timestamp_rows=ties,
                         source_sha256=source_sha256,
                         reference_bytes=book.nbytes + times.nbytes, encoded_bytes=replay.bytes(),
                         checkpoints=len(replay.checkpoints), replay_sha256=digest,
                         snapshot_mismatches=0))
    totals = {key: sum(row[key] for row in rows)
              for key in ("updates", "equal_timestamp_rows", "reference_bytes",
                          "encoded_bytes", "checkpoints", "snapshot_mismatches")}
    totals["reduction_percent"] = 100 * (1 - totals["encoded_bytes"] / totals["reference_bytes"])
    result = {"levels": FIELDS // 4, "checkpoint_stride": CHECKPOINT,
              "timestamp_unit": "picoseconds since midnight", "symbols": rows, "total": totals}
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
