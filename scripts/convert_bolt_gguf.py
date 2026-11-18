#!/usr/bin/env python3
"""Convert nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 to the llama.cpp gemma4 GGUF contract.

Reads the main safetensors shard, writes a Q8_0 GGUF with the canonical
tensor names of the llama.cpp gemma4 reference (src/models/gemma4.cpp,
src/llama-arch.cpp):

  - scalar metadata and arrays per BOLT_INT_SCALARS / FLOAT_SCALARS /
    check_bolt_layer_pattern (sliding pattern 5+1; kv heads 2 on every
    layer — shared layers 24..41 read their source layers' KV, and a
    zero head count made the fork SIGFPE in ggml_flash_attn_ext);
  - per-layer embeddings as ONE per_layer_token_embd table
    [PER_LAYER_INPUT * LAYERS, VOCAB] in the reference layout: each row is
    one token's 42 layer blocks contiguous (ne0 = PLE*LAYERS), the HF
    embed_tokens_per_layer table verbatim — element (layer 3, token 7,
    feature 5) sits at row 7, column 3*256 + 5;
  - the vocabulary embedded from tokenizer.json (tokens, scores,
    token_type; merges too when the model is BPE);
  - the frozen shared-KV file and the shared layers' k/v projections are
    NOT converted (serving reads the source layers' KV);
  - rope_freqs is 256 ones: the resident kernel's built-in global mode
    (base 1e6 on the first 64 pairs, zero after) already matches the
    fork's partial rotary 0.25, so the divisor table is neutral.

Two passes: the plan pass reads only the safetensors header, the write
pass lifts one tensor to fp32 at a time (bf16 lifts exactly by bit
shift), so peak memory stays at the largest single tensor.

Usage: convert_bolt_gguf.py <model_dir> <out.gguf>
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

LAYERS = 42
SHARED = 18
PHYSICAL = LAYERS - SHARED
HIDDEN = 2560
FEED_FORWARD = 10240
QUERY_HEADS = 8
KV_HEADS = 2
VOCAB = 262144
LOCAL_HEAD = 256
GLOBAL_HEAD = 512
PER_LAYER_INPUT = 256
# Tokens per merged-table write chunk (see cols_all): small enough to bound
# peak RSS, a free variable so the selfcheck can cross chunk edges.
CHUNK_TOKENS = 4096

GGUF_F32 = 0
GGUF_Q80 = 8


class Shard:
    """Header-indexed mmap reader; BF16 lifts to F32 exactly by bit shift."""

    def __init__(self, path: Path):
        with open(path, "rb") as handle:
            count = struct.unpack("<Q", handle.read(8))[0]
            self.header = json.loads(handle.read(count))
        self.header.pop("__metadata__", None)
        self.base = 8 + count
        self.view = np.memmap(path, dtype=np.uint8, mode="r")

    def shape(self, name: str) -> list:
        try:
            return self.header[name]["shape"]
        except KeyError:
            raise SystemExit(f"missing tensor {name}") from None

    def get(
        self, name: str, row: int | None = None, cols: tuple[int, int] | None = None,
        rows: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """The tensor, one leading-dim row of it, or a column slice of a 2-d
        BF16 tensor, lifted to F32."""
        info = self.header[name]
        start, end = info["data_offsets"]
        shape = info["shape"]
        if row is not None:
            if len(shape) != 3:
                raise SystemExit(f"{name}: row fetch needs a 3-d tensor")
            row_len = (end - start) // shape[0]
            start, end = start + row * row_len, start + (row + 1) * row_len
            shape = shape[1:]
        raw = self.view[self.base + start : self.base + end]
        if info["dtype"] == "BF16":
            wide = raw.view(np.uint16)
            if rows is not None:
                if len(shape) != 2:
                    raise SystemExit(f"{name}: row-range fetch needs a 2-d tensor")
                wide = wide.reshape(shape)[rows[0] : rows[1]]
                shape = [rows[1] - rows[0], shape[1]]
            if cols is not None:
                if len(shape) != 2:
                    raise SystemExit(f"{name}: column fetch needs a 2-d tensor")
                wide = wide.reshape(shape)[:, cols[0] : cols[1]]
                shape = [shape[0], cols[1] - cols[0]]
            lifted = wide.astype(np.uint32) << 16
            return lifted.view(np.float32).reshape(shape)
        if info["dtype"] == "F32":
            if cols is not None:
                raise SystemExit(f"{name}: column fetch needs BF16")
            return raw.view(np.float32).reshape(shape)
        raise SystemExit(f"{name}: unsupported dtype {info['dtype']}")


def sized(text: bytes) -> bytes:
    return struct.pack("<Q", len(text)) + text


def kv_string(key: str, value: str) -> bytes:
    return sized(key.encode()) + struct.pack("<I", 8) + sized(value.encode())


def kv_u32(key: str, value: int) -> bytes:
    return sized(key.encode()) + struct.pack("<I", 4) + struct.pack("<I", value)


def kv_f32(key: str, value: float) -> bytes:
    return sized(key.encode()) + struct.pack("<I", 6) + struct.pack("<f", value)


def kv_bool(key: str, value: bool) -> bytes:
    return sized(key.encode()) + struct.pack("<I", 7) + bytes([int(value)])


def kv_i32_array(key: str, values) -> bytes:
    out = sized(key.encode()) + struct.pack("<I", 9) + struct.pack("<I", 5)
    out += struct.pack("<Q", len(values))
    return out + b"".join(struct.pack("<i", v) for v in values)


def kv_bool_array(key: str, values) -> bytes:
    out = sized(key.encode()) + struct.pack("<I", 9) + struct.pack("<I", 7)
    out += struct.pack("<Q", len(values))
    return out + bytes(int(v) for v in values)


def kv_string_array(key: str, values) -> bytes:
    out = sized(key.encode()) + struct.pack("<I", 9) + struct.pack("<I", 8)
    out += struct.pack("<Q", len(values))
    return out + b"".join(sized(v.encode()) for v in values)


def kv_f32_array(key: str, values) -> bytes:
    out = sized(key.encode()) + struct.pack("<I", 9) + struct.pack("<I", 6)
    out += struct.pack("<Q", len(values))
    return out + b"".join(struct.pack("<f", v) for v in values)


def kv_u32_array(key: str, values) -> bytes:
    out = sized(key.encode()) + struct.pack("<I", 9) + struct.pack("<I", 4)
    out += struct.pack("<Q", len(values))
    return out + b"".join(struct.pack("<I", v) for v in values)


BYTE_TOKEN = __import__("re").compile(r"^<0x[0-9A-Fa-f]{2}>$")


def load_vocabulary(model_dir: Path):
    """(tokens, scores, types, merges) in id order, from tokenizer.json.

    llama.cpp token types: 1 normal, 2 unknown, 3 control, 4 user_defined,
    5 unused, 6 byte. BPE scores are all 0.0 and the merge list travels
    beside them; the gemma4 reference loader requires the merges, so a
    tokenizer without them is refused here instead of failing at load.
    """
    tok = json.loads((model_dir / "tokenizer.json").read_text())
    model = tok["model"]
    merges = model.get("merges", [])
    if model["type"] != "BPE" or not merges:
        why = "not BPE" if model["type"] != "BPE" else "BPE without merges"
        raise SystemExit(
            f"tokenizer.json is {why}; the gemma4 reference loader requires a BPE vocabulary with merges"
        )
    vocab = {}
    scores = {}
    for token, index in model["vocab"].items():
        vocab[token] = int(index)
    for entry in tok.get("added_tokens", []):
        vocab[entry["content"]] = int(entry["id"])
        scores.setdefault(entry["content"], 0.0)
    size = max(vocab.values()) + 1
    tokens = [None] * size
    score_list = [0.0] * size
    type_list = [1] * size
    for token, index in vocab.items():
        tokens[index] = token
        score_list[index] = scores.get(token, 0.0)
        type_list[index] = 6 if BYTE_TOKEN.match(token) else 1
    for entry in tok.get("added_tokens", []):
        type_list[vocab[entry["content"]]] = 3 if entry.get("special") else 4
    if any(t is None for t in tokens):
        raise SystemExit(f"tokenizer vocabulary has a gap at id {tokens.index(None)}")
    merges = model.get("merges", [])
    if merges and isinstance(merges[0], list):
        merges = [" ".join(pair) for pair in merges]
    return tokens, score_list, type_list, merges


def quantize_q8_0(flat: np.ndarray) -> bytes:
    """Q8_0 blocks of 32: f16 scale then 32 int8 codes, along the flat layout."""
    if flat.shape[0] % 32:
        raise SystemExit(f"element count {flat.shape[0]} not a multiple of 32")
    blocks = flat.reshape(-1, 32)
    amax = np.abs(blocks).max(axis=1)
    scale = (amax / 127.0).astype(np.float16)
    safe = np.where(scale == 0, np.float16(1.0), scale)
    codes = np.rint(blocks / safe[:, None])
    codes = np.clip(codes, -127, 127).astype(np.int8)
    out = np.empty((blocks.shape[0], 34), dtype=np.uint8)
    out[:, :2] = scale.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = codes.view(np.uint8).reshape(-1, 32)
    return out.tobytes()


def payload_bytes(weight: np.ndarray, ggml_type: int) -> bytes:
    if ggml_type == GGUF_F32:
        return weight.astype(np.float32).tobytes()
    return quantize_q8_0(weight.reshape(-1))


def payload_len(ggml_type: int, dims) -> int:
    elements = 1
    for dim in dims:
        elements *= dim
    if ggml_type == GGUF_F32:
        return elements * 4
    if ggml_type == GGUF_Q80:
        if elements % 32:
            raise SystemExit(
                f"Q8_0 needs a multiple of 32 elements, got {elements}"
            )
        return elements // 32 * 34
    raise SystemExit(f"unhandled type {ggml_type}")


def build_plan(shard: Shard):
    """(gguf name, ggml type, dims [ne0, ne1], source) per emitted tensor.

    source: ('hf', name) | ('row', name, layer) | ('ones', count).
    """
    plan = []
    mat, f32 = GGUF_Q80, GGUF_F32

    def check(name: str, shape) -> None:
        found = shard.shape(name)
        if list(found) != list(shape):
            raise SystemExit(f"{name} has shape {found}, expected {list(shape)}")

    def emit(name, ty, dims, source):
        plan.append((name, ty, dims, source))

    check("model.language_model.embed_tokens.weight", [VOCAB, HIDDEN])
    emit("token_embd.weight", mat, [HIDDEN, VOCAB], ("hf", "model.language_model.embed_tokens.weight"))
    check("model.language_model.norm.weight", [HIDDEN])
    emit("output_norm.weight", f32, [HIDDEN], ("hf", "model.language_model.norm.weight"))
    emit("rope_freqs.weight", f32, [LOCAL_HEAD], ("ones", LOCAL_HEAD))
    check(
        "model.language_model.per_layer_model_projection.weight",
        [LAYERS * PER_LAYER_INPUT, HIDDEN],
    )
    emit(
        "per_layer_model_proj.weight",
        mat,
        [HIDDEN, LAYERS * PER_LAYER_INPUT],
        ("hf", "model.language_model.per_layer_model_projection.weight"),
    )
    check("model.language_model.per_layer_projection_norm.weight", [PER_LAYER_INPUT])
    emit(
        "per_layer_proj_norm.weight",
        f32,
        [PER_LAYER_INPUT],
        ("hf", "model.language_model.per_layer_projection_norm.weight"),
    )
    tables = "model.language_model.embed_tokens_per_layer.weight"
    table_shape = shard.shape(tables)
    if table_shape != [VOCAB, LAYERS * PER_LAYER_INPUT]:
        raise SystemExit(
            f"{tables} has shape {table_shape}, expected"
            f" [vocab, {LAYERS * PER_LAYER_INPUT}]"
        )

    for layer in range(LAYERS):
        head = LOCAL_HEAD if layer % 6 != 5 else GLOBAL_HEAD
        hf = f"model.language_model.layers.{layer}."
        gguf = f"blk.{layer}."
        physical = layer < PHYSICAL

        def norm(suffix, hf_suffix, width):
            name = hf + hf_suffix
            check(name, [width])
            emit(gguf + suffix, f32, [width], ("hf", name))

        def matrix(suffix, hf_suffix, out_dim, in_dim):
            name = hf + hf_suffix
            check(name, [out_dim, in_dim])
            emit(gguf + suffix, mat, [in_dim, out_dim], ("hf", name))

        norm("attn_norm.weight", "input_layernorm.weight", HIDDEN)
        matrix("attn_q.weight", "self_attn.q_proj.weight", QUERY_HEADS * head, HIDDEN)
        norm("attn_q_norm.weight", "self_attn.q_norm.weight", head)
        if physical:
            matrix("attn_k.weight", "self_attn.k_proj.weight", KV_HEADS * head, HIDDEN)
            norm("attn_k_norm.weight", "self_attn.k_norm.weight", head)
            matrix("attn_v.weight", "self_attn.v_proj.weight", KV_HEADS * head, HIDDEN)
        matrix("attn_output.weight", "self_attn.o_proj.weight", HIDDEN, QUERY_HEADS * head)
        norm("post_attention_norm.weight", "post_attention_layernorm.weight", HIDDEN)
        norm("ffn_norm.weight", "pre_feedforward_layernorm.weight", HIDDEN)
        matrix("ffn_gate.weight", "mlp.gate_proj.weight", FEED_FORWARD, HIDDEN)
        matrix("ffn_up.weight", "mlp.up_proj.weight", FEED_FORWARD, HIDDEN)
        matrix("ffn_down.weight", "mlp.down_proj.weight", HIDDEN, FEED_FORWARD)
        norm("post_ffw_norm.weight", "post_feedforward_layernorm.weight", HIDDEN)
        matrix("inp_gate.weight", "per_layer_input_gate.weight", PER_LAYER_INPUT, HIDDEN)
        matrix("proj.weight", "per_layer_projection.weight", HIDDEN, PER_LAYER_INPUT)
        norm("post_norm.weight", "post_per_layer_input_norm.weight", HIDDEN)
        check(hf + "layer_scalar", [1])
        emit(gguf + "layer_output_scale.weight", f32, [1], ("hf", hf + "layer_scalar"))

    emit(
        "per_layer_token_embd.weight",
        mat,
        [PER_LAYER_INPUT * LAYERS, VOCAB],
        ("cols_all", tables),
    )
    return plan


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    model_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    shard = Shard(model_dir / "model-00001-of-00001.safetensors")

    plan = build_plan(shard)
    heads = [KV_HEADS] * LAYERS
    pattern = [layer % 6 != 5 for layer in range(LAYERS)]
    tokens, tok_scores, tok_types, merges = load_vocabulary(model_dir)

    kv_entries = [
        kv_string("general.architecture", "gemma4"),
        kv_u32("gemma4.block_count", LAYERS),
        kv_u32("gemma4.context_length", 65536),
        kv_u32("gemma4.embedding_length", HIDDEN),
        kv_u32("gemma4.embedding_length_per_layer_input", PER_LAYER_INPUT),
        kv_u32("gemma4.feed_forward_length", FEED_FORWARD),
        kv_u32("gemma4.attention.head_count", QUERY_HEADS),
        kv_u32("gemma4.attention.key_length", GLOBAL_HEAD),
        kv_u32("gemma4.attention.key_length_swa", LOCAL_HEAD),
        kv_u32("gemma4.attention.value_length", GLOBAL_HEAD),
        kv_u32("gemma4.attention.value_length_swa", LOCAL_HEAD),
        kv_u32("gemma4.attention.sliding_window", 512),
        kv_u32("gemma4.attention.shared_kv_layers", SHARED),
        kv_u32("gemma4.rope.dimension_count", GLOBAL_HEAD),
        kv_u32("gemma4.rope.dimension_count_swa", LOCAL_HEAD),
        kv_f32("gemma4.rope.freq_base", 1_000_000.0),
        kv_f32("gemma4.rope.freq_base_swa", 10_000.0),
        kv_f32("gemma4.final_logit_softcapping", 30.0),
        kv_f32("gemma4.attention.layer_norm_rms_epsilon", 1.0e-6),
        kv_string("tokenizer.ggml.model", "gemma4"),
        kv_string_array("tokenizer.ggml.tokens", tokens),
        kv_f32_array("tokenizer.ggml.scores", tok_scores),
        kv_i32_array("tokenizer.ggml.token_type", tok_types),
        kv_string_array("tokenizer.ggml.merges", merges) if merges else b"",
        kv_bool("tokenizer.ggml.add_bos_token", True),
        kv_u32("tokenizer.ggml.bos_token_id", 2),
        kv_u32("tokenizer.ggml.eos_token_id", 1),
        kv_u32("tokenizer.ggml.unknown_token_id", 3),
        kv_i32_array("gemma4.attention.head_count_kv", heads),
        kv_bool_array("gemma4.attention.sliding_window_pattern", pattern),
        kv_i32_array("tokenizer.ggml.suppress_tokens", [258883, 258882]),
    ]
    kv_entries = [entry for entry in kv_entries if entry]

    infos = []
    offset = 0
    for name, ty, dims, _ in plan:
        infos.append((name, ty, dims, offset))
        offset = -(-(offset + payload_len(ty, dims)) // 32) * 32

    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", len(plan))
    header += struct.pack("<Q", len(kv_entries))
    for entry in kv_entries:
        header += entry
    for name, ty, dims, tensor_offset in infos:
        header += sized(name.encode())
        header += struct.pack("<I", len(dims))
        for dim in dims:
            header += struct.pack("<Q", dim)
        header += struct.pack("<I", ty)
        header += struct.pack("<Q", tensor_offset)

    print(f"{len(plan)} tensors, header {len(header)} bytes, payload {offset} bytes", flush=True)
    with open(out_path, "wb") as out:
        out.write(bytes(header))
        out.write(b"\0" * (-len(header) % 32))
        for index, (name, ty, dims, source) in enumerate(plan):
            kind = source[0]
            if kind == "ones":
                payload = payload_bytes(np.ones(source[1], np.float32), ty)
            elif kind == "hf":
                payload = payload_bytes(shard.get(source[1]), ty)
            elif kind == "cols":
                start = source[2]
                payload = payload_bytes(
                    shard.get(source[1], cols=(start, start + PER_LAYER_INPUT)), ty
                )
            elif kind == "cols_all":
                # The HF table is already [VOCAB, PLE*LAYERS]: token-major with
                # every layer's block contiguous per token — the reference
                # layout. Stream it in token chunks, quantising each chunk
                # straight to the output: lifting the whole table at once
                # would peak well past the host's free memory.
                chunk_tokens = CHUNK_TOKENS
                written = 0
                for start in range(0, VOCAB, chunk_tokens):
                    stop = min(start + chunk_tokens, VOCAB)
                    piece = payload_bytes(
                        shard.get(source[1], rows=(start, stop)), ty
                    )
                    out.write(piece)
                    written += len(piece)
                    if (start // chunk_tokens) % 16 == 0:
                        print(f"  table {stop}/{VOCAB} tokens", flush=True)
                expected = payload_len(ty, dims)
                if written != expected:
                    raise SystemExit(f"{name}: streamed {written} != expected {expected}")
                out.write(b"\0" * (-written % 32))
                continue
            else:
                payload = payload_bytes(shard.get(source[1], row=source[2]), ty)
            expected = payload_len(ty, dims)
            if len(payload) != expected:
                raise SystemExit(f"{name}: payload {len(payload)} != expected {expected}")
            out.write(payload)
            out.write(b"\0" * (-len(payload) % 32))
            if index % 50 == 0 or index == len(plan) - 1:
                print(f"  {index + 1}/{len(plan)} written", flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
