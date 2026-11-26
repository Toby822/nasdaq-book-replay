"""Write and open a memory-mapped snapshot replay archive."""
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from .replay import Replay

HEADER = struct.Struct("<8sI")
MAGIC = b"LOBREPL1"
ARRAYS = ("times", "pointers", "columns", "deltas", "checkpoints")


def save_mapped(book, path):
    path = Path(path)
    arrays = [np.ascontiguousarray(getattr(book, name)) for name in ARRAYS]
    metadata = json.dumps({"stride": book.stride,
                           "arrays": [(name, array.dtype.str, array.shape)
                                      for name, array in zip(ARRAYS, arrays)]}).encode()
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            tmp = Path(file.name)
            file.write(HEADER.pack(MAGIC, len(metadata)))
            file.write(metadata)
            for array in arrays:
                file.write(memoryview(array).cast("B") if array.size else b"")
            file.flush()
            os.fsync(file.fileno())
        tmp.replace(path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def open_mapped(path):
    path = Path(path)
    with path.open("rb") as file:
        magic, length = HEADER.unpack(file.read(HEADER.size))
        if magic != MAGIC or not 1 <= length <= 4096:
            raise ValueError("Invalid replay header")
        metadata = json.loads(file.read(length))
    offset = HEADER.size + length
    arrays = {}
    for name, dtype, shape in metadata["arrays"]:
        dtype = np.dtype(dtype)
        count = int(np.prod(shape))
        end = offset + count * dtype.itemsize
        if end > path.stat().st_size:
            raise ValueError("Truncated replay archive")
        arrays[name] = np.memmap(path, mode="r", offset=offset, shape=tuple(shape), dtype=dtype)
        offset = end
    if tuple(arrays) != ARRAYS or offset != path.stat().st_size:
        raise ValueError("Invalid replay layout")
    return Replay(**arrays, stride=metadata["stride"])
