"""Print file offsets for the two original packed tensors of Inkling blk.5."""
import struct
import sys
from pathlib import Path


def offsets(path):
    expected = {
        "blk.5.ffn_gate_up_exps.weight": (16, (2304, 2304, 128), 256, 66),
        "blk.5.ffn_down_exps.weight": (42, (1152, 2304, 128), 64, 18),
    }
    scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with path.open("rb") as stream:
        def read(fmt):
            fmt = struct.Struct("<" + fmt)
            return fmt.unpack(stream.read(fmt.size))[0]

        def string():
            return stream.read(read("Q")).decode()

        def skip(kind):
            if kind in scalar:
                stream.seek(scalar[kind], 1)
            elif kind == 8:
                stream.seek(read("Q"), 1)
            elif kind == 9:
                element, count = read("I"), read("Q")
                if element in scalar:
                    stream.seek(scalar[element] * count, 1)
                else:
                    for _ in range(count):
                        skip(element)
            else:
                raise ValueError(f"unexpected GGUF metadata type {kind}")

        if stream.read(4) != b"GGUF" or read("I") != 3:
            raise ValueError("expected GGUF version 3")
        tensors, metadata = read("Q"), read("Q")
        alignment = 32
        for _ in range(metadata):
            key, kind = string(), read("I")
            if key == "general.alignment" and kind == 4:
                alignment = read("I")
            else:
                skip(kind)
        if alignment == 0 or alignment & (alignment - 1):
            raise ValueError("invalid GGUF alignment")

        found = {}
        for _ in range(tensors):
            name = string()
            shape = tuple(read("Q") for _ in range(read("I")))
            kind, offset = read("I"), read("Q")
            if name in expected:
                typ, dims, block, block_bytes = expected[name]
                if (kind, shape) != (typ, dims) or offset % alignment or name in found:
                    raise ValueError(f"unexpected descriptor for {name}: {kind}, {shape}, {offset}")
                cols, rows, experts = shape
                found[name] = (offset, cols // block * block_bytes * rows * experts)
        data_start = (stream.tell() + alignment - 1) // alignment * alignment
        size = path.stat().st_size
        result = []
        for name in expected:
            offset, nbytes = found[name]
            if data_start + offset + nbytes > size:
                raise ValueError(f"truncated tensor {name}")
            result.append(data_start + offset)
        return result


if __name__ == "__main__":
    print(*offsets(Path(sys.argv[1])))
