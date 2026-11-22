#!/usr/bin/env python3
"""Extract the three exact Inkling Q8_0 projection matrices, without repacking."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct


# GGUF orders dimensions as (input columns, output rows).
TENSORS = {
    "blk.0.attn_k.weight": (2304, 1024),
    "blk.1.attn_gate.weight": (2304, 3072),
    "blk.0.attn_output.weight": (3072, 2304),
}
SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Reader:
    def __init__(self, stream):
        self.stream = stream
        self.size = os.fstat(stream.fileno()).st_size

    def read(self, size):
        if size > self.size - self.stream.tell():
            raise ValueError("truncated GGUF")
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("short GGUF read")
        return data

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def name(self):
        size = self.number("Q")
        if size > 1024:
            raise ValueError("oversized GGUF descriptor name")
        return self.read(size).decode("utf-8")

    def skip(self, count):
        if count > self.size - self.stream.tell():
            raise ValueError("metadata span exceeds GGUF length")
        self.stream.seek(count, os.SEEK_CUR)

    def skip_value(self, kind):
        if kind in SCALAR_BYTES:
            self.skip(SCALAR_BYTES[kind])
        elif kind == 8:
            self.skip(self.number("Q"))
        elif kind == 9:
            element, count = self.number("I"), self.number("Q")
            if element in SCALAR_BYTES:
                self.skip(SCALAR_BYTES[element] * count)
            elif element == 8:
                if count > (self.size - self.stream.tell()) // 8:
                    raise ValueError("invalid GGUF string array length")
                for _ in range(count):
                    self.skip(self.number("Q"))
            else:
                raise ValueError(f"unsupported GGUF array element type {element}")
        else:
            raise ValueError(f"unsupported GGUF metadata type {kind}")


def alignment(reader, metadata):
    value = None
    for _ in range(metadata):
        key, kind = reader.name(), reader.number("I")
        if key != "general.alignment":
            reader.skip_value(kind)
            continue
        if kind != 4 or value is not None:
            raise ValueError("malformed or duplicate general.alignment")
        value = reader.number("I")
        if value == 0 or value & (value - 1):
            raise ValueError("invalid GGUF alignment")
    return 32 if value is None else value


def descriptors(reader):
    if reader.read(4) != b"GGUF" or reader.number("I") != 3:
        raise ValueError("expected little-endian GGUF version 3")
    n_tensors, metadata = reader.number("Q"), reader.number("Q")
    align = alignment(reader, metadata)
    found = {}
    for _ in range(n_tensors):
        name, rank = reader.name(), reader.number("I")
        if not 1 <= rank <= 4:
            raise ValueError(f"bad tensor rank for {name}: {rank}")
        dims = tuple(reader.number("Q") for _ in range(rank))
        kind, offset = reader.number("I"), reader.number("Q")
        if name not in TENSORS:
            continue
        if kind != 8 or dims != TENSORS[name] or offset % align or name in found:
            raise ValueError(f"unexpected Q8_0 descriptor for {name}: {kind}, {dims}, {offset}")
        cols, rows = dims
        found[name] = (offset, cols // 32 * 34 * rows)
    if missing := TENSORS.keys() - found.keys():
        raise ValueError(f"missing tensors: {', '.join(sorted(missing))}")
    data_start = (reader.stream.tell() + align - 1) // align * align
    last_end = data_start
    for name, (offset, size) in sorted(found.items(), key=lambda kv: kv[1][0]):
        if data_start + offset < last_end or data_start + offset + size > reader.size:
            raise ValueError(f"overlapping or truncated tensor {name}")
        last_end = data_start + offset + size
    return data_start, found


def extract(checkpoint, output):
    with checkpoint.open("rb") as stream:
        reader = Reader(stream)
        data_start, found = descriptors(reader)
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"gguf": str(checkpoint.resolve()), "gguf_bytes": reader.size, "weights": []}
        for name, (cols, rows) in TENSORS.items():
            relative, size = found[name]
            offset = data_start + relative
            stream.seek(offset)
            data = reader.read(size)
            filename = f"q8-{cols}-{rows}.bin"
            (output / filename).write_bytes(data)
            manifest["weights"].append({"tensor": name, "codec": 8, "cols": cols, "rows": rows,
                                        "offset": offset, "bytes": size, "file": filename,
                                        "sha256": hashlib.sha256(data).hexdigest()})
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, help="new directory for raw tensors and SHA-256 manifest")
    args = parser.parse_args()
    print(json.dumps(extract(args.checkpoint, args.output), indent=2))
