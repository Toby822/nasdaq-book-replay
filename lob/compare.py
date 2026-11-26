"""Measure checkpoint storage versus deterministic random-seek decode work."""
import json
from pathlib import Path
from time import perf_counter_ns
from tempfile import TemporaryDirectory

import numpy as np

from .mapped import open_mapped, save_mapped
from .replay import Replay, SYMBOLS, read_sample

STRIDES = (512, 2048, 8192)


def compare():
    rng = np.random.default_rng(42)
    results = []
    with TemporaryDirectory() as tmp:
        for symbol in SYMBOLS:
            time, events, snapshots = read_sample(Path("data") / (symbol + ".zip"))
            queries = time[rng.integers(0, len(time), 1000)]
            positions = np.searchsorted(time, queries, side="right") - 1
            for stride in STRIDES:
                path = Path(tmp) / (symbol + ".lob")
                save_mapped(Replay.encode(time, snapshots, stride=stride), path)
                replay = open_mapped(path)
                for timestamp, pos in zip(queries[:50], positions[:50]):
                    np.testing.assert_array_equal(replay.as_of(timestamp), snapshots[pos])
                measured = []
                for _ in range(3):
                    started = perf_counter_ns()
                    for timestamp in queries:
                        replay.as_of(timestamp)
                    measured.append((perf_counter_ns() - started) / (1000 * len(queries)))
                results.append({"symbol": symbol, "checkpoint_rows": stride,
                                "archive_bytes": path.stat().st_size,
                                "median_rows_decoded_per_lookup": float(np.median(positions % stride + 1)),
                                "median_warm_mean_us": float(np.median(measured))})
                del replay
    result = {"queries_per_symbol": 1000, "warm_runs": 3, "per_symbol": results}
    (Path("results") / "checkpoint_sweep.json").write_text(json.dumps(result, indent=2) + "\n")
    for stride in STRIDES:
        group = [r for r in results if r["checkpoint_rows"] == stride]
        print(stride, sum(r["archive_bytes"] for r in group),
              [r["median_rows_decoded_per_lookup"] for r in group],
              [round(r["median_warm_mean_us"], 1) for r in group])


if __name__ == "__main__":
    compare()
