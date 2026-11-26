"""Benchmark the mapped format and reconcile every saved row with its source."""
import hashlib
import json
from pathlib import Path

from .mapped import open_mapped, save_mapped
from .replay import Replay, SYMBOLS, read_sample, verify


def main():
    output = Path("results")
    output.mkdir(exist_ok=True)
    results = []
    for symbol in SYMBOLS:
        times, events, depth = read_sample(Path("data") / f"{symbol}.zip")
        encoded = Replay.encode(times, depth)
        path = output / f"{symbol}.lob"
        save_mapped(encoded, path)
        mapped = open_mapped(path)
        digest, ties = verify(mapped, depth, events)
        # Stream in file order and independently check a digest across checkpoint boundaries.
        streamed = hashlib.sha256()
        for row in mapped.iter_rows(0, len(depth)):
            streamed.update(row.astype("<i4", copy=False).tobytes())
        assert digest == streamed.hexdigest()
        results.append({"symbol": symbol, "rows": len(depth), "tied_timestamp_rows": ties,
                        "source_sha256": hashlib.sha256((Path("data") / f"{symbol}.zip").read_bytes()).hexdigest(),
                        "dense_bytes": times.nbytes + depth.nbytes, "file_bytes": path.stat().st_size,
                        "mismatches": 0, "depth_sha256": digest})
    dense = sum(row["dense_bytes"] for row in results)
    stored = sum(row["file_bytes"] for row in results)
    summary = {"session_date": "2012-06-21", "levels": 10, "checkpoint_rows": 2048,
               "timestamp_unit": "picoseconds since midnight",
               "symbols": results, "rows": sum(row["rows"] for row in results),
               "dense_bytes": dense, "file_bytes": stored,
               "reduction_percent": 100 * (1 - stored / dense)}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
