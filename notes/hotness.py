#!/usr/bin/env python3
"""Analytical hotness estimate for quantize_row_q8_K from a GGUF file.

Counts, per token and per prefill batch, how many activation elements are fed to
quantize_row_q8_K through the matmul activation packing path:

  for every weight tensor whose vec_dot_type is Q8_K, one activation row of
  ne[0] elements is quantized per src1 row. So per token the count is
  sum(ne[0]) over those tensors, and per prefill batch it is n_tokens times that
  (each of the 512 rows is quantized separately).

Types whose vec_dot_type is Q8_K are all K-quants and all IQ K-quants
(ggml/src/ggml-cpu/ggml-cpu.c:311-386). On this CPU the repack selector
(ggml/src/ggml-cpu/repack.cpp:4970-5145) only repacks Q4_0, Q4_K, IQ4_NL and
MXFP4 (all 8x8 with AVX2), and repacked tensors use their own activation
quantizer, so those are excluded in the default configuration and included with
--no-repack.

Usage: hotness.py MODEL.gguf [MODEL2.gguf ...]
"""

import struct
import sys

# ggml/src/ggml-quants.h / ggml.h
TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0", 43: "F8_E4M3",
}

# vec_dot_type == GGML_TYPE_Q8_K for these (ggml-cpu.c type traits table)
Q8K_TYPES = {
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS",
    "IQ1_S", "IQ3_S", "IQ2_S", "IQ4_XS", "IQ1_M", "TQ1_0", "TQ2_0",
}

# repacked on this machine (AVX2, no AVX-512/NEON/SVE/VXE/RVV)
REPACKED_TYPES = {"Q4_0", "Q4_K", "IQ4_NL", "MXFP4"}

QK_K = 256

_FIXED = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
          5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
          12: ("<d", 8)}


def read_gguf(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file: %s" % path)
        version, = struct.unpack("<I", f.read(4))
        n_tensors, = struct.unpack("<Q", f.read(8))
        n_kv, = struct.unpack("<Q", f.read(8))

        def rstr():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et, = struct.unpack("<I", f.read(4))
                n, = struct.unpack("<Q", f.read(8))
                return [rval(et) for _ in range(n)]
            fmt, size = _FIXED[t]
            return struct.unpack(fmt, f.read(size))[0]

        kv = {}
        for _ in range(n_kv):
            k = rstr()
            t, = struct.unpack("<I", f.read(4))
            kv[k] = rval(t)

        tensors = []
        for _ in range(n_tensors):
            name = rstr()
            nd, = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            ttype, = struct.unpack("<I", f.read(4))
            struct.unpack("<Q", f.read(8))  # offset
            tensors.append((name, TYPE_NAMES.get(ttype, "TYPE_%d" % ttype), dims))
    return version, kv, tensors


def report(path):
    version, kv, tensors = read_gguf(path)
    print("== %s" % path)
    arch = kv.get("general.architecture", "?")
    n_layer = kv.get("%s.block_count" % arch, None)
    n_embd = kv.get("%s.embedding_length" % arch, None)
    print("   gguf v%d arch=%s block_count=%s embedding_length=%s tensors=%d"
          % (version, arch, n_layer, n_embd, len(tensors)))

    q8k = [t for t in tensors if t[1] in Q8K_TYPES]
    q8k_repacked = [t for t in q8k if t[1] in REPACKED_TYPES]
    q8k_plain = [t for t in q8k if t[1] not in REPACKED_TYPES]

    def elems_per_token(ts, per_layer_only=True):
        # matmul weights are 2D; the contraction dim is ne[0]
        total = 0
        for name, ttype, dims in ts:
            if len(dims) != 2:
                continue
            if per_layer_only and not name.startswith("blk."):
                continue
            total += dims[0]
        return total

    all_e = elems_per_token(q8k)
    plain_e = elems_per_token(q8k_plain)
    repack_e = elems_per_token(q8k_repacked)
    head_e = sum(dims[0] for name, ttype, dims in q8k if len(dims) == 2 and not name.startswith("blk."))

    print("   Q8_K-activation tensors: %d (of which repacked on this CPU: %d)"
          % (len(q8k), len(q8k_repacked)))
    print("   elements quantized per token (blocks + output head):")
    print("     --no-repack / all K+IQ tensors : %10d elements  (%d blocks)" % (all_e + head_e, (all_e + head_e) // QK_K))
    print("     default build (repack active)  : %10d elements  (%d blocks)"
          % (plain_e + head_e, (plain_e + head_e) // QK_K))
    print("     (excluded because repacked     : %10d elements)" % repack_e)
    print("   elements quantized per 512-token prefill batch:")
    print("     --no-repack : %12d elements" % ((all_e + head_e) * 512))
    print("     default     : %12d elements" % ((plain_e + head_e) * 512))

    per_type = {}
    for name, ttype, dims in tensors:
        if len(dims) == 2:
            per_type[ttype] = per_type.get(ttype, 0) + 1
    print("   weight tensor type histogram: %s"
          % ", ".join("%s=%d" % (k, v) for k, v in sorted(per_type.items())))
    return all_e + head_e, plain_e + head_e


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for p in sys.argv[1:]:
        report(p)
