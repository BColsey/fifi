#!/usr/bin/env python3
"""End-to-end self-check of scripts/convert_bolt_gguf.py at tiny dims.

Fabricates a mini BF16 safetensors shard + tokenizer.json with every
tensor name the plan checks, runs the converter (constants monkeypatched
down), parses the output GGUF, and asserts the canonical contract:
the six reference names, no legacy names, the merged per-layer table,
head_count_kv with no zeros, and the embedded vocabulary.
"""

import importlib.util
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
OUT = Path(tempfile.mkdtemp(prefix="bolt-selfcheck-"))
MODEL = OUT / "model"
MODEL.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("conv", str(HERE / "convert_bolt_gguf.py"))
assert spec is not None and spec.loader is not None, "converter module must load"
conv: Any = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

# Tiny dims: every emitted element count must stay a multiple of 32.
conv.LAYERS = 6
conv.SHARED = 2
conv.PHYSICAL = conv.LAYERS - conv.SHARED
conv.HIDDEN = 64
conv.FEED_FORWARD = 128
conv.QUERY_HEADS = 2
conv.KV_HEADS = 1
conv.VOCAB = 96
conv.LOCAL_HEAD = 32
conv.GLOBAL_HEAD = 64
conv.PER_LAYER_INPUT = 16
# Cross chunk edges: 96 tokens in chunks of 40 gives 40/40/16 with a
# partial last chunk, exercising the streaming path's edge handling.
conv.CHUNK_TOKENS = 40
L, H, PLE, V = conv.LAYERS, conv.HIDDEN, conv.PER_LAYER_INPUT, conv.VOCAB

tensors = {}


def put(name, shape, fill):
    tensors[name] = (list(shape), np.full(shape, fill, np.float32).astype("<f4").view(np.uint8).view(np.uint16) >> 0)


def bf16_bytes(fill, shape):
    f32 = np.full(shape, fill, np.float32)
    # round-trip through the converter's own BF16 lift convention: store
    # the upper 16 bits, which the Shard class shifts back left.
    return (f32.view(np.uint32) >> 16).astype("<u2").tobytes()


payloads = {}
header = {}
offset = 0


def add(name, shape, fill):
    global offset
    data = bf16_bytes(fill, shape)
    header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
    payloads[name] = data
    offset += len(data)


add("model.language_model.embed_tokens.weight", [V, H], 0.5)
add("model.language_model.norm.weight", [H], 1.0)
add("model.language_model.per_layer_model_projection.weight", [L * PLE, H], 0.25)
add("model.language_model.per_layer_projection_norm.weight", [PLE], 1.0)
def table_fill(t, c):
    # Distinct per (token, 32-wide block): constant blocks quantize exactly,
    # so each Q8 block's f16 scale d uniquely names its source block.
    return np.float32(1000.0 + t * 100.0 + (c // 32))


table = np.fromfunction(table_fill, (V, L * PLE), dtype=np.int64).astype(np.float32)
add(
    "model.language_model.embed_tokens_per_layer.weight",
    [V, L * PLE],
    table,
)

for layer in range(L):
    head = conv.LOCAL_HEAD if layer % 6 != 5 else conv.GLOBAL_HEAD
    base = f"model.language_model.layers.{layer}."
    physical = layer < conv.PHYSICAL
    add(base + "input_layernorm.weight", [H], 1.0)
    add(base + "self_attn.q_proj.weight", [conv.QUERY_HEADS * head, H], 0.1)
    add(base + "self_attn.q_norm.weight", [head], 1.0)
    if physical:
        add(base + "self_attn.k_proj.weight", [conv.KV_HEADS * head, H], 0.1)
        add(base + "self_attn.k_norm.weight", [head], 1.0)
        add(base + "self_attn.v_proj.weight", [conv.KV_HEADS * head, H], 0.1)
    add(base + "self_attn.o_proj.weight", [H, conv.QUERY_HEADS * head], 0.1)
    add(base + "post_attention_layernorm.weight", [H], 1.0)
    add(base + "pre_feedforward_layernorm.weight", [H], 1.0)
    add(base + "mlp.gate_proj.weight", [conv.FEED_FORWARD, H], 0.2)
    add(base + "mlp.up_proj.weight", [conv.FEED_FORWARD, H], 0.2)
    add(base + "mlp.down_proj.weight", [H, conv.FEED_FORWARD], 0.2)
    add(base + "post_feedforward_layernorm.weight", [H], 1.0)
    add(base + "per_layer_input_gate.weight", [PLE, H], 0.3)
    add(base + "per_layer_projection.weight", [H, PLE], 0.3)
    add(base + "post_per_layer_input_norm.weight", [H], 1.0)
    add(base + "layer_scalar", [1], 2.0)

raw_header = json.dumps(header).encode()
with open(MODEL / "model-00001-of-00001.safetensors", "wb") as fh:
    fh.write(struct.pack("<Q", len(raw_header)))
    fh.write(raw_header)
    if len(raw_header) % 8:
        fh.write(b"\0" * (8 - len(raw_header) % 8))
    for name in header:
        fh.write(payloads[name])

tokenizer = {
    "model": {
        "type": "BPE",
        "vocab": {
            "<pad>": 0,
            "<unk>": 1,
            "he": 2,
            "llo": 3,
            "wor": 4,
            "ld": 5,
            "<0x41>": 6,
        },
        # pair form: exercises the converter's pair->"a b" normalization
        "merges": [["he", "llo"], ["wor", "ld"]],
    },
    "added_tokens": [
        {"id": 7, "content": "<s>", "special": True},
        {"id": 8, "content": "[MASK]", "special": False},
    ],
}
(MODEL / "tokenizer.json").write_text(json.dumps(tokenizer))

sys.argv = ["convert_bolt_gguf.py", str(MODEL), str(OUT / "mini.gguf")]
conv.main()

# ---- parse the output GGUF header ----
blob = (OUT / "mini.gguf").read_bytes()


def u32(buf, at):
    return struct.unpack_from("<I", buf, at)[0], at + 4


def u64(buf, at):
    return struct.unpack_from("<Q", buf, at)[0], at + 8


def sized_bytes(buf, at):
    n, at = u64(buf, at)
    return buf[at : at + n], at + n


assert blob[:4] == b"GGUF"
at = 8
tensor_count, at = u64(blob, at)
kv_count, at = u64(blob, at)
kv = {}
array_types = {}
for _ in range(kv_count):
    key, at = sized_bytes(blob, at)
    vtype, at = u32(blob, at)
    if vtype == 4:  # u32
        val, at = u32(blob, at)
    elif vtype == 6:  # f32
        val = struct.unpack_from("<f", blob, at)[0]
        at += 4
    elif vtype == 7:  # bool
        val = blob[at]
        at += 1
    elif vtype == 8:  # string
        val, at = sized_bytes(blob, at)
        val = val.decode()
    elif vtype == 9:  # array
        atype, at = u32(blob, at)
        n, at = u64(blob, at)
        items = []
        for _ in range(n):
            if atype == 4:
                item, at = u32(blob, at)
            elif atype == 5:
                item = struct.unpack_from("<i", blob, at)[0]
                at += 4
            elif atype == 6:
                item = struct.unpack_from("<f", blob, at)[0]
                at += 4
            elif atype == 7:
                item = blob[at]
                at += 1
            elif atype == 8:
                item, at = sized_bytes(blob, at)
                item = item.decode()
            else:
                raise SystemExit(f"array type {atype} not handled")
            items.append(item)
        array_types[key.decode()] = atype
        val = items
    else:
        raise SystemExit(f"value type {vtype} not handled")
    kv[key.decode()] = val

names = []
for _ in range(tensor_count):
    name, at = sized_bytes(blob, at)
    ndims, at = u32(blob, at)
    dims = []
    for _ in range(ndims):
        d, at = u64(blob, at)
        dims.append(d)
    _, at = u32(blob, at)  # ggml type
    tensor_offset, at = u64(blob, at)
    names.append((name.decode(), dims, tensor_offset))

failures = []
want_names = [
    "per_layer_model_proj.weight",
    "per_layer_proj_norm.weight",
    "blk.0.inp_gate.weight",
    "blk.0.proj.weight",
    "blk.0.post_norm.weight",
    "per_layer_token_embd.weight",
]
flat = {n for n, _, _ in names}
for want in want_names:
    if want not in flat:
        failures.append(f"missing canonical tensor {want}")
for legacy in [
    "per_layer_model_projection.weight",
    "per_layer_projection_norm.weight",
    "blk.0.per_layer_gate.weight",
    "blk.0.per_layer_proj.weight",
    "blk.0.post_per_layer_input_norm.weight",
    "blk.0.per_layer_embd.weight",
]:
    if legacy in flat:
        failures.append(f"legacy tensor {legacy} still present")

merged = [dims for n, dims, _ in names if n == "per_layer_token_embd.weight"]
if not merged or merged[0] != [PLE * L, V]:
    failures.append(f"per_layer_token_embd dims {merged}, want [{PLE * L}, {V}]")

# Pin the reference layout: token-major, every layer's 256-feature block
# contiguous per token. Block i is token t=i//336, source column block b=i%336
# (layer L owns b in [L*8, L*8+8); feature 5 of layer 3, token 7 is block 7*336+24).
data_start = (at + 31) & ~31  # tensor offsets are relative to the aligned header end
_, _, merged_offset = next(n for n in names if n[0] == "per_layer_token_embd.weight")
blocks_per_row = (PLE * L) // 32
for i in range(0, V * blocks_per_row, 37):  # sample stride keeps it cheap
    t, b = divmod(i, blocks_per_row)
    block_at = data_start + merged_offset + i * 34
    d = struct.unpack_from("<e", blob, block_at)[0]
    want = (1000.0 + t * 100.0 + b) / 127.0
    if abs(d - want) > 0.3:
        failures.append(f"merged block {i} (token {t}, block {b}): d={d}, want {want}")
        break

heads = kv.get("gemma4.attention.head_count_kv")
if heads != [conv.KV_HEADS] * L:
    failures.append(f"head_count_kv {heads}, want all {conv.KV_HEADS} (no zeros)")

tok = kv.get("tokenizer.ggml.tokens")
types = kv.get("tokenizer.ggml.token_type")
if tok is None or len(tok) != 9:
    failures.append(f"tokens {tok}")
elif types is None:
    failures.append("token_type is missing")
else:
    if types[tok.index("<0x41>")] != 6:
        failures.append("byte token not typed 6")
    if types[tok.index("<s>")] != 3:
        failures.append("special token not typed 3")
    if types[tok.index("[MASK]")] != 4:
        failures.append("user_defined token not typed 4")
if "gemma4.embedding_length_per_layer_input" not in kv:
    failures.append("missing KV gemma4.embedding_length_per_layer_input")
if kv.get("tokenizer.ggml.model") != "gemma4":
    failures.append(f"tokenizer.ggml.model {kv.get('tokenizer.ggml.model')!r}, want 'gemma4'")
if array_types.get("tokenizer.ggml.token_type") != 5:
    failures.append(
        f"token_type array type {array_types.get('tokenizer.ggml.token_type')}, want i32 (5)"
    )
if kv.get("tokenizer.ggml.merges") != ["he llo", "wor ld"]:
    failures.append(f"merges {kv.get('tokenizer.ggml.merges')!r}, want the two normalized pairs")

if failures:
    for line in failures:
        print("FAIL:", line)
    raise SystemExit(1)
print(f"SELFCHECK OK: {tensor_count} tensors, {kv_count} kv entries, merged table {[PLE * L, V]}")
shutil.rmtree(OUT, ignore_errors=True)
