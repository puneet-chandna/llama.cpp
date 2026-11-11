# Phase 2: research notes and candidate selection

All file:line references are to this tree at `b5cb8f6f4` (branch `ai/simd-experiment`).

## 1. How the arch fallback mechanism works, and what is actually scalar on this machine

### 1.1 Mechanism

The generic C implementations live in two "shared" sources that are compiled for every architecture:

- `ggml/src/ggml-cpu/quants.c` (has every `*_generic` quantizer and dot product; e.g. `quantize_row_q8_K_generic` at :126 which just calls `quantize_row_q8_K_ref`)
- `ggml/src/ggml-cpu/repack.cpp` (generic repack gemv/gemm kernels)

Both include `ggml/src/ggml-cpu/arch-fallback.h` (`quants.c:10`, `repack.cpp:12`). That header, per architecture, defines `#define X_generic X`, so the generic definition *overwrites* the arch symbol name and becomes the selected implementation. The arch-specific bodies live in separate translation units added by `ggml/src/ggml-cpu/CMakeLists.txt:284-285` (`ggml-cpu/arch/x86/quants.c`, `ggml-cpu/arch/x86/repack.cpp`). `ggml/src/ggml-cpu/arch/x86/` contains exactly three files: `cpu-feats.cpp`, `quants.c`, `repack.cpp`.

Branch structure of `arch-fallback.h` and the line of each branch head:

| line | condition |
|---|---|
| 7 | `GGML_CPU_GENERIC` |
| 78 | `__aarch64__` / `__arm__` / MSVC ARM |
| **88** | `__x86_64__` / `__i386__` / `_M_IX86` / `_M_X64` |
| 118 | `__POWERPC__` |
| 169 | `__loongarch64` |
| 220 | `__riscv` |
| 264 | `__s390x__` |
| 317 | `__wasm__` |

Correction to the plan's audit: the x86 branch is lines 88-117, not "~118-149".

`GGML_CPU_GENERIC` is defined in exactly one place, `ggml/src/ggml-cpu/CMakeLists.txt:623`, inside the final `else()` of the architecture dispatch ("Unknown CPU architecture. Falling back to generic implementations."). It is therefore **not** defined in this build (x86_64 with `-DGGML_NATIVE=ON`), and the `quantize_row_q8_K_generic -> quantize_row_q8_K` rename at `arch-fallback.h:11` is inactive. The other three occurrences of that rename (lines 121, 171, 266) belong to the powerpc, loongarch64 and s390x branches.

The x86 branch (lines 88-117) remaps **only** repack symbols: `ggml_quantize_mat_q8_0_4x4`, `ggml_quantize_mat_q8_K_4x4`, and `gemv`/`gemm` for `q1_0_4x4/4x8`, `q4_0_4x4/4x8`, `q4_K_8x4`, `q5_K_8x4/8x8`, `q6_K_8x4/8x8`, `iq4_nl_4x4`, `mxfp4_4x4`, `q8_0_4x4/4x8`. It does not touch `quantize_row_q8_K_generic`, so on x86 both symbols exist and are distinct.

Re-confirmed in the built artifact (stop condition 1):

```
$ nm -C --defined-only build/release/bin/libggml-cpu.so | grep quantize_row_q8_K
00000000001106a0 T quantize_row_q8_K
0000000000059800 T quantize_row_q8_K_generic

$ objdump -d --no-show-raw-insn -M intel build/release/bin/libggml-cpu.so | awk '/<quantize_row_q8_K>:/,/ret/'
00000000001106a0 <quantize_row_q8_K>:
  1106a0:	endbr64
  1106a4:	jmp    155a0 <quantize_row_q8_K_ref@plt>
  1106a9:	nop    DWORD PTR [rax+0x0]
```

`quantize_row_q8_K` is the symbol registered in the type-traits table (`ggml/src/ggml-cpu/ggml-cpu.c:402` -> `.from_float = quantize_row_q8_K`), so every `from_float` call site listed below executes the scalar reference. `quantize_row_q8_K_generic` is never referenced.

### 1.2 Reachability audit: which x86-generic kernels are actually used here

Feature state on this CPU (`-march=native`): AVX2 yes, AVX-512 no, NEON no, SVE no, VXE no, RISC-V V no, AMX no.

`ggml_repack_get_optimal_repack_type` (`ggml/src/ggml-cpu/repack.cpp:4970-5145`) is the only gate for repacking. Non-null results on this machine:

| src0 type | line | condition | result |
|---|---|---|---|
| Q4_0 | 4980 | `ggml_cpu_has_avx2()` and `ne[1] % 8 == 0` | `q4_0_8x8_q8_0` |
| Q4_K | 5012 | `ggml_cpu_has_avx2()` and `ne[1] % 8 == 0` | `q4_K_8x8_q8_K` |
| IQ4_NL | 5078 | `ggml_cpu_has_avx2()` and `ne[1] % 8 == 0` | `iq4_nl_8x8_q8_0` |
| MXFP4 | 5100 | `ggml_cpu_has_avx2()` and `ne[1] % 8 == 0` | `mxfp4_8x8_q8_0` |
| Q2_K | 5039 | `ggml_cpu_has_avx512()` | false here |
| Q5_K | 5056 | NEON + `matmul_int8` or NEON + `dotprod` | false here |
| Q6_K | 5067 | NEON + `matmul_int8` or NEON + `dotprod` | false here |
| Q8_0, Q1_0 | 5111, 5133 | NEON (+ RVV 16x1) | false here |
| IQ4_XS | - | no case at all | never repacked |

Consequences:

- The `gemv`/`gemm` `q5_K_8x8` and `q6_K_8x8` kernels that `arch-fallback.h:98-100,111-113` binds to their generic implementations in `ggml/src/ggml-cpu/repack.cpp:1118` are **never selected on x86** - dead code here, not a hot fallback. This matches the plan's audit.
- Repacked tensors use their own activation quantizer: `ggml_quantize_mat_q8_K_4x8` at `ggml/src/ggml-cpu/arch/x86/repack.cpp:290` has its own `#if defined(__AVX2__)` body over `block_q8_Kx4`. So the Q4_K/Q4_0/IQ4_NL/MXFP4 tensors of a Q4_K_M model bypass the candidate entirely, while **IQ4_XS (all tensors), Q2_K, Q5_K, Q6_K and Q3_K do not**.
- `--no-repack` / `-nr` (`common/arg.cpp:2395`) forces every tensor through `quantize_row_q8_K`, which makes the candidate cover every K/IQ matmul.

### 1.3 The candidate and its callers

`ggml/src/ggml-cpu/arch/x86/quants.c:504-507`:

```c
// placeholder implementation for Apple targets
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}
```

There is a second disabled in-tree SIMD variant: `quantize_row_q8_K_vnni` at `ggml/src/ggml-cpu/amx/mmq.cpp:392`, an AVX-512 implementation, whose *only* call site is switched off:

```c
// ggml/src/ggml-cpu/amx/mmq.cpp:469-478
template <>
inline void from_float<block_q8_K>(const float * x, char * vy, int64_t k) {
#if 1
    // TODO: this is reference impl!
    quantize_row_q8_K_ref(x, (block_q8_K *)vy, k);
#else
    quantize_row_q8_K_vnni(x, vy, k);
#endif
}
```

So even the AMX path uses the scalar reference today. Note that `quantize_row_q8_K_vnni` is *not* bit-compatible with the reference either: it uses `iscale = 127.f / amax` (positive) and stores `q = cvt(round(v * iscale))` without the `MIN(127, v)` clamp, i.e. the opposite sign convention for `qs` relative to `d`. AMX is absent on this CPU, so this path is unreachable regardless.

Call sites that reach the candidate (all obtain `from_float` from the traits table):

| site | context | when |
|---|---|---|
| `ggml/src/ggml-cpu/ggml-cpu.c:1359,1383` | `ggml_compute_forward_mul_mat` activation packing | once per src1 row per op |
| `ggml/src/ggml-cpu/ggml-cpu.c:1639,1652` | `ggml_compute_forward_mul_mat_id` (MoE) | once per src1 row per op |
| `ggml/src/ggml-cpu/tiled/tiled.cpp:1041` (fetch), `:1074` (call) | tiled matmul, rows >= 8 | once per src1 row per op |
| `ggml/src/ggml-cpu/ops.cpp:8891` | `ggml_compute_forward_flash_attn_ext` when K is a K-quant | once per KV row |

Tiled matmul is enabled by default; `GGML_CPU_TILED_MM` (read at `ggml/src/ggml-cpu/tiled/tiled.cpp:577`) is the master switch (`enabled = env == NULL || atoi(env) != 0`), and `GGML_CPU_TILED_MM_FORCE` (`:588`) forces it even when unprofitable. Both are useful knobs for the "too good" cross-check in phase 5.

### 1.4 The scalar set that is *not* a fallback

These are single implementations compiled for every architecture; `arch-fallback.h` contains no `rope`, `norm`, `softmax` or `bf16` entry at all (grep returns nothing):

- RoPE: `ggml_compute_forward_rope_flt` (`ggml/src/ggml-cpu/ops.cpp:6295`) has no `GGML_SIMD` block and no `ggml_vec_*` call inside its element loop.
- RMS norm (`ops.cpp`) accumulates the sum of squares with `ggml_float` (`sum += (ggml_float)(x[i00] * x[i00]);`) and then calls `ggml_vec_scale_f32`, which comes from `ggml/src/ggml-cpu/vec.h` - a single shared header (listed at `ggml/src/ggml-cpu/CMakeLists.txt:55`) whose `GGML_SIMD` bodies are compiled on all architectures.
- `ggml_cpu_fp32_to_bf16` is defined once, `ggml/src/ggml-cpu/ggml-cpu.c:3598`, with no arch/x86 counterpart.

Optimising these would not be "filling in a missing arch kernel" and they are each a small share of runtime, so they are out of scope.

## 2. Candidate table

| candidate | (a) real-world usage | (b) layout/algorithm simplicity | (c) expected kernel speedup | (d) prior art | verdict |
|---|---|---|---|---|---|
| **`quantize_row_q8_K`** | vec_dot_type of Q2_K/Q3_K/Q4_K/Q5_K/Q6_K and every IQ K-quant (`ggml-cpu.c:311-386`, `.from_float` at `:402`). Hit by every matmul of an IQ quant, by q6_K tensors of a Q4_K_M model, and by every matmul under `-nr`. Also FA with quantized KV. | Simple: abs-max scan, one scale, round-to-nearest, sum of 16 bytes. No codebook, no bit-plane packing, no extra scale stages. `block_q8_K` is 292 bytes with `qs` at offset 4. | High: the reference is a tight scalar loop with a 256-iteration abs-max pass and a second 256-iteration loop doing `nearest_int` (magic-add) plus a 256-iteration bsum loop. All three vectorise well under AVX2. | Upstream PR 22331 (state `open`, created 2026-04-24, title "ggml-cpu: optimize q8 quantization on x86 SIMD") targets exactly this function. Fork experiment only; not read, not copied. In-tree `quantize_row_q8_K_vnni` is disabled. | **chosen** |
| `ggml_gemv_q6_K_8x8_q8_K`, `ggml_gemm_q6_K_8x8_q8_K`, `q5_K_8x8` | Would be the repack path for Q5_K/Q6_K | Complex (interleaved 8x8 blocks, LUT unpack) | Reported roughly pp +16.8% / tg -2.9% in PR 19706 | PRs 19706 and 19707 both `open` (created 2026-02-18, titles "Q6_K - Block Interleaving Implementation for x86 SIMD (AVX512/AVX2)" and "Q5_K - ...") | rejected: **unreachable here** - the selector needs NEON (`repack.cpp:5056-5075`), so no run on this machine would exercise it |
| `ggml_cpu_fp32_to_bf16` | conversion of F32 activations to BF16 tensors | Trivial (truncate/shift) | High per-call | none in tree | rejected: only reachable for F16/BF16 models, and a 7B BF16 model (14 GB) does not fit in this machine's ~5 GB `MemAvailable`; a 1B BF16 model would be the only exercisable case, and it is not in the model family this experiment standardised on |
| RoPE / RMS-norm / softmax / binary-unary / im2col / conv / ssm | everywhere | varies | low each | none - single implementation on all arches | rejected: not a fallback, see 1.4; each is a small share of runtime |
| Q2_K repack selector `ggml_cpu_has_avx512()` (`repack.cpp:5039`) | would repack Q2_K | - | - | an AVX2 `q2_K_8x8_q8_K` implementation exists in `ggml/src/ggml-cpu/arch/arm/repack.cpp`? no - x86 has no q2_K 8x8 body, only generic | observation only (see 4.2): the AVX-512 gate excludes AVX2 machines that could run the *generic* kernel, but the generic kernel is not faster than the non-repack path, so this is a gating inconsistency, not a proven perf bug |

## 3. Chosen candidate: `quantize_row_q8_K` on x86

### 3.1 Block layout

`ggml/src/ggml-common.h` (QK_K = 256):

```c
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
```

0-63 bytes are produced by this kernel. `ggml/src/ggml-cpu/tiled/tiled-kernel.cpp:563-565` asserts the exact offsets the tiled path relies on:

```c
static_assert(sizeof(block_q8_K) == 292 && offsetof(block_q8_K, qs) == 4,
              "block_q8_K layout changed, fix the src1 repack");
```

So `sizeof == 292`, `d` at 0, `qs` at 4 (`bsums` at 4+256 = 260). Every access to `qs`/`bsums` must be unaligned.

### 3.2 The reference algorithm, quoted

`ggml/src/ggml-quants.c:2784-2816`:

```c
void quantize_row_q8_K_ref(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}
```

and `ggml/src/ggml-quants.c:630-635`:

```c
static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
```

Exactness-relevant details, in the order they matter:

1. **argmax with a first-wins rule.** `amax` starts at `+0.0f` and is updated only on a *strict* `ax > amax`. So `max` is `x[j*]` where `j*` is the **first** index maximising `|x[j]|`. The tie rule is not cosmetic: `iscale = -127.f/max` depends on the **sign** of `max`, so when the block contains both `+v` and `-v` with `|v|` maximal, the choice of `j*` flips the sign of `iscale` and therefore every byte in `qs`. Ties in `|x|` with opposite signs are not exotic - they occur for every all-zero-mean symmetric block, and they occur in practice in normalised activations.
2. **`if (!amax)`.** `!amax` is true when `amax == +0.0f`. This is the case for an all-zero block **and for a block whose values are all NaN**, because `fabsf(NaN) = NaN` and `NaN > 0` is false, so `amax` is never updated. In this branch the reference writes `d = 0`, zero-fills `qs` only, and **leaves `bsums` untouched** (callers must not assume `bsums` is initialised). It does not touch `bsums` in any way.
3. **`iscale = -127.f/max`**, using the signed `max`, not `amax`. `d = 1/iscale` is stored after the quantisation loop (relevant only because it is the last write to the block).
4. **Rounding.** `nearest_int` is round-half-to-**even**, implemented with the magic-add trick: for `|fval| <= 2^22 - 1` the bit pattern of `fval + 1.5*2^23` encodes `round_to_nearest_even(fval)` in its low 23 bits, biased by `2^22`.
5. **Clamp.** `MIN(127, v)` with no lower clamp. The assignment to `int8_t` truncates modulo 256, which is implementation-defined in C++ but is what the current toolchains do; it is only reachable when `v > 127` or `v < -128`.
6. **`bsums[j] = sum of qs[16j .. 16j+15]`**, which for `|qs| <= 127` is an int16 quantity with `|bsums[j]| <= 2032` (the `int sum` accumulator is int32 and cannot overflow).
7. **`nb = k / QK_K`, elements past `nb*QK_K` are ignored.** The caller contract is `k % 256 == 0` (asserted) but the reference simply ignores a tail if the assert is compiled out.

### 3.3 Where the reference is slow

Per block: 256 iterations of `fabsf` + compare + two conditional moves; 256 iterations of `nearest_int` (float add, `movd`, mask, subtract, `MIN`) plus an `int8_t` store; 16x16 int8 accumulations. That is roughly 1500-2000 instructions per 256 elements, all scalar, with a data-dependent branch pattern in loop 1 and one store per byte in loop 2. It is the only quantizer in `arch/x86/quants.c` without an AVX2 body.

### 3.4 Closest existing SIMD kernels (structure reference only)

- `quantize_row_q8_0` (`ggml/src/ggml-cpu/arch/x86/quants.c:302`) and `quantize_row_q8_1` (`:400`): the local style reference. They show the `andnot` + `max_ps` abs-max tree, the `cvtps_epi32` -> `packs_epi32` -> `packs_epi16` -> `permutevar8x32_epi32` 32-float-to-32-int8 packing, and the `hsum` helpers. They are also the empirical proof for the tolerance rule used in phase 3: the reference-vs-AVX2 diff for `quantize_row_q8_0` is 0 bytes on every corpus.
- `ggml_quantize_mat_q8_K_4x8` (`ggml/src/ggml-cpu/arch/x86/repack.cpp:290`): computes `Q[_][i] = nearest_int(iscale[_]*x[_][i])` for 4 rows at once with its own AVX2 code, including a 32-element packing loop. Useful as a second style reference; it is *not* the same kernel (different output layout `block_q8_Kx4`, no `bsums`). It does not share code, so no refactor is implied.
- `quantize_row_q8_K_vnni` (`ggml/src/ggml-cpu/amx/mmq.cpp:392`): AVX-512, disabled, and not bit-compatible (positive `iscale`, no clamp, `dpbusd`-based bsums). Read for structure only.

### 3.5 Existing wasm / RISC-V `quantize_row_q8_K`: correct reasons they are not usable as oracles

Both are the only other SIMD `q8_K` quantizers in the tree and both diverge from the reference; the plan's parenthetical reason was partly wrong, so the verified reasons are:

- `ggml/src/ggml-cpu/arch/wasm/quants.c:145`: argmax is computed as `float amax = -min > max ? min : max;` from `wasm_f32x4_pmax`/`pmin` reductions, i.e. a *global* max/min comparison, not the reference's first-wins argmax. Rounding is `wasm_f32x4_nearest` = round-half-to-even (**matches** `nearest_int`), and narrowing uses `wasm_i16x8_narrow_i32x4` / `wasm_i8x16_narrow_i16x8`, which **saturate** at both ends instead of `MIN(127, v)` + modulo-256 truncation. Its zero case (`:172-177`) writes `d = 0` and zero-fills `qs`, leaving `bsums` untouched, matching the reference.
- `ggml/src/ggml-cpu/arch/riscv/quants.c:122`: `amax = fabsf(max_val) > fabsf(min_val) ? ... ` (same global-comparison tie rule), rounding is `vfcvt_x_f_v_i32m8_rm(RNE)` (round-half-to-even, **matches**), narrowing is `vnclip` with `RNE`, which **saturates**. Its zero case explicitly `memset(y_block->bsums, 0, sizeof(y_block->bsums))`, which **differs** from the reference.

So the divergence is the argmax tie rule, the out-of-range clamp convention (saturating narrow vs `MIN(127, .)` plus int8 conversion), and for RISC-V the `bsums` handling - not the rounding mode. Neither can be used as a semantic reference; the scalar reference is the only oracle.

### 3.6 Planned instructions

| intrinsic | instruction | role |
|---|---|---|
| `_mm256_loadu_ps` / `_mm256_storeu_si256` | `VMOVUPS` / `VMOVDQU` | unaligned access; required because `qs` sits at byte offset 4 inside a 292-byte struct |
| `_mm256_andnot_ps` | `VANDNPS` | `|x|` by clearing the sign bit (Intel SDM Vol. 2A) |
| `_mm256_cmp_ps` with `_CMP_GT_OQ` | `VCMPPS` | ordered, non-signalling `>`; false for NaN, which is exactly the reference's `ax > amax` behaviour. `_mm256_max_ps` is **not** usable for the argmax: `VMAXPS` returns the second operand if either is NaN, so it propagates NaN and would break the all-NaN and mixed-NaN cases |
| `_mm256_blendv_ps` / `_mm256_blendv_epi8` | `VBLENDVPS` / `VPBLENDVB` | conditional update of the per-lane maximum and of the per-lane first-hit vector index |
| `_mm256_mul_ps` | `VMULPS` | `iscale*x[j]` |
| `_mm256_cvtps_epi32` | `VCVTPS2DQ` | round according to MXCSR (default round-to-nearest-**even**), which is what makes it equal to `nearest_int` for the reachable range |
| `_mm256_min_epi32` | `VPMINSD` | the `MIN(127, v)` clamp |
| `_mm256_packs_epi32` + `_mm256_packs_epi16` | `VPACKSSDW` + `VPACKSSWB` | int32 -> int16 -> int8; saturation is unreachable because `v` is already clamped to `<= 127` and bounded below by `>= -127` (see 3.8) |
| `_mm256_permute4x64_epi64(..., 0xD8)` | `VPERMQ` | fix the `VPACKSSWB` 128-bit lane interleave; 0xD8 = qword order (0,2,1,3), identical to the `permutevar8x32_epi32(0,4,1,5,2,6,3,7)` used by `quantize_row_q8_0` |
| `_mm256_maddubs_epi16` | `VPMADDUBSW` | pair sums for `bsums`: **first operand must be the unsigned constant `1`**, second operand the signed `q`, so each int16 is `q[2i] + q[2i+1]` |
| `_mm256_madd_epi16` | `VPMADDWD` | widen pair sums to int32, giving sums of 4 consecutive bytes |
| `_mm256_hadd_epi32` | `VPHADDD` | two steps: 4-byte sums -> 8-byte sums -> 16-byte sums |
| `_mm256_permutevar8x32_epi32` | `VPERMD` | gather the 16-byte sums into store order |
| `_mm256_cmp_ps` with `_CMP_EQ_OQ`, `_mm256_movemask_ps`, `TZCNT` | `VCMPPS`/`VMOVMSKPS`/`TZCNT` (BMI1) | variant B only: locate the first index equal to the block maximum. BMI1 is present on this CPU |

Optimisation rationale (Intel 64 and IA-32 Optimization Reference Manual): `VPMADDUBSW`/`VPMADDWD` are single-uop on port 0/1/5 with 0.5-1 cycle throughput on this microarchitecture, and replacing 16 scalar int8 loads and adds per `bsums` entry with a two-instruction reduction removes 256 byte loads per block and the associated store-to-load forwarding chain.

### 3.7 Variant A: exact per-lane first-argmax, then requantise

Pass 1 (32 iterations of 8 floats) keeps two `__m256`/`__m256i` states per lane:

```
axv    = andnot(signbit, v)                          // |x|
gt     = cmp_ps(axv, amaxv, _CMP_GT_OQ)              // ordered >, false for NaN
amaxv  = blendv_ps(amaxv, axv, gt)
kbestv = blendv_epi8(kbestv, set1_epi32(it), gt)     // first vector index per lane
```

Then fold lanes 0..7 into scalars:

```
float amax = 0.0f; int bestk = INT_MAX; int best = -1;
for (int L = 0; L < 8; ++L) {
    const float ax = axlane[L];
    const int   kb = kbest[L];
    if (ax > amax || (ax == amax && kb < bestk)) { amax = ax; bestk = kb; best = kb*8 + L; }
}
```

Why element index `kb*8 + L` is the global first argmax: lane `L` at iteration `it` holds element `8*it + L`. Comparing two candidate elements `(kbA, LA)` and `(kbB, LB)` by `kb` first and lane order second is the same order as comparing `8*kb + L` directly, because `8*kbA + LA` and `8*kbB + LB` compare in `kb` first, and on `kbA == kbB` in `LA`. The lane loop visits lanes in increasing order and replaces only on a strictly smaller `kb`, which resolves `kbA == kbB` in favour of the smaller lane.

The signed value is recovered as `max = x[best]`, which is *by construction* the same expression the reference stores (`max = x[j]` at the first argmax), so the sign/tie rule cannot drift. This removes the `maxv` blend from the inner loop (one fewer op and one fewer live register than the plan's formulation) while making the equivalence stronger, not weaker.

`_CMP_GT_OQ` is mandatory. `_mm256_max_ps` has NaN-propagating semantics that do not match `ax > amax`, and the comparison must be ordered so NaN never updates a lane - exactly like the reference, where a NaN `ax` never satisfies `ax > amax`.

Degenerate guard (cold in practice, but must delegate to the reference to be exact):

```
if (!(amax > 0.0f && isfinite(max) && isfinite(iscale))) { quantize_row_q8_K_ref(x, &y[i], QK_K); continue; }
```

`amax == 0` (all-zero, all-NaN, and NaN-only-maximum blocks) reproduces the reference's `if (!amax)` branch verbatim, including the untouched `bsums`. `!isfinite(max)` covers any Inf in the block. `!isfinite(iscale)` covers maxima so small that `-127.f/max` overflows to Inf.

**Correction to the plan's guard.** The plan specifies `!(amax >= FLT_MIN && amax <= FLT_MAX)`. That is not sufficient. Take a block of 256 copies of `1.0e-37f`: `amax = 1e-37f` is a *normal* float (`FLT_MIN = 1.17549435e-38`), so the plan's guard does not fire, but `iscale = -127.f/1e-37f = -1.27e39` overflows to `-Inf` (`FLT_MAX = 3.4e38`). The reference then evaluates `nearest_int(-Inf * 1e-37f) = nearest_int(-Inf)`: `-Inf + 12582912.f = -Inf`, bit pattern `0xFF800000`, `(0xFF800000 & 0x007FFFFF) - 0x00400000 = -4194304`, `MIN(127, -4194304) = -4194304`, and the store to `int8_t` keeps the low byte `0x00`, i.e. **`qs[j] = 0`** and `d = 1/-Inf = -0.0f`. An AVX2 `VCVTPS2DQ(-Inf)` returns the invalid-operation sentinel `0x80000000`, which after `VPMINSD`/`VPACKSSDW`/`VPACKSSWB` becomes **`-128`**. That is a real byte difference, reachable with finite inputs, and the plan's own contingency says to fix such a case by widening the guard rather than by adding a tolerance. Hence the guard above also tests `isfinite(iscale)`.

Range invariant used by pass 2. With `amax` finite, `amax > 0` and `iscale` finite we have `|iscale| = 127/|max| = 127/amax` up to one rounding, and `|x[j]| <= amax`, so `|iscale*x[j]| <= 127*(1 + 2^-23)`. Rounding to nearest therefore lands in `[-127, 127]`, which means `MIN(127, v)` is a no-op at the top and the reference's modulo-256 `int8_t` conversion never wraps at the bottom. `VPACKSSDW`/`VPACKSSWB` saturation is consequently unreachable too, so the pack chain implements the reference's store exactly. The `_mm256_min_epi32(vi, 127)` clamp is kept anyway: one cheap op per vector that makes the correspondence to `MIN(127, v)` literal and keeps the code robust if the bound argument is ever revised.

Pass 2, 32 elements per iteration, 8 iterations:

```
vi      = cvtps_epi32(mul(loadu_ps(x + off), iscale_vec))
vi      = min_epi32(vi, 127)
a       = packs_epi32(vi0, vi1)
b       = packs_epi32(vi2, vi3)
packed  = permute4x64_epi64(packs_epi16(a, b), 0xD8)
storeu_si256(y[i].qs + 32*j, packed)
```

Lane derivation for the pack+permute chain (32 elements, `viN` holds elements `8N..8N+7`):

| after | qword0 | qword1 | qword2 | qword3 |
|---|---|---|---|---|
| `packs_epi16(a,b)` | elems 0-3, 8-11 | elems 16-19, 24-27 | elems 4-7, 12-15 | elems 20-23, 28-31 |
| after `permute4x64(0xD8)` = (0,2,1,3) | elems 0-7 | elems 8-15 | elems 16-23 | elems 24-31 |

`bsums` for 64 elements per iteration (this is where the plan's "one 32-byte store" is refined into a verified 4-store form):

```
t0   = maddubs_epi16(ones_u8, q0)     // q[2i] + q[2i+1]        -> int16, 16 values
t1   = maddubs_epi16(ones_u8, q1)
t0   = madd_epi16(t0, ones_i16)       // 4-byte sums as int32    -> 8 values
t1   = madd_epi16(t1, ones_i16)
g    = hadd_epi32(t0, t1)             // 8-byte sums: lane0 = [S0(0..7), S0(8..15), S1(0..7), S1(8..15)], lane1 = [S0(16..23), S0(24..31), S1(16..23), S1(24..31)]
g    = hadd_epi32(g, g)               // 16-byte sums: [b0, b2, b0, b2 | b1, b3, b1, b3]
g    = permutevar8x32_epi32(g, {0,4,1,5,0,4,1,5})   // [b0, b1, b2, b3, b0, b1, b2, b3]
g    = packs_epi32(g, g)              // int16 pairs; low 8 bytes = b0,b1,b2,b3
storel_epi64(y[i].bsums + 4*jj, castsi256_si128(g))
```

with `b0..b3 = bsums[4*jj .. 4*jj+3]` (elements `64*jj .. 64*jj+63`). Bounds: `|q[2i]+q[2i+1]| <= 254` (int16, no saturation), `|4-byte sum| <= 508` (int32), `|8-byte| <= 1016`, `|16-byte| <= 2032` - all far inside int16/int32, so no intermediate can saturate or overflow, and the final int16 values are bit-identical to the reference's `int sum` accumulator narrowed to `int16_t`.

Cost per block: 12 stores (`8 * 32 B` for `qs` + `4 * 8 B` for `bsums`), no scalar extract, no byte-at-a-time store.

### 3.8 Variant B: reduction-first argmax plus an exact repair pass

Pass 1 becomes `andnot` + `max_ps` (2 ops per vector, no compare, no blend, no index tracking), then one horizontal max. To recover the reference's *first* argmax index an extra pass over the 32 vectors is needed, comparing `|x| == amax` (`_CMP_EQ_OQ`, safe because `amax > 0` and the values compared are `|x|`), taking `movemask_ps` and `tzcnt`, and exiting at the first hit. Pass 2 and `bsums` are shared with variant A.

Variant B trades `3` ops per vector in pass 1 (compare, 2 blends) for roughly `5` ops per vector in a second pass that stops at the first match. Both variants are exact; the choice is made by measurement (`simd-bench --mode single --k 4096,32768`) and the loser is deleted.

### 3.9 Variant C

Unrolling pass 2 to 4 independent 64-element groups with 2-4 independent accumulator chains for `bsums`. Keep only if faster; the final tree must contain exactly one implementation with no dead alternatives.

## 4. Observations to report (not proven bugs)

### 4.1 wasm and RISC-V `quantize_row_q8_K` diverge from the scalar reference

Evidence in 3.5. Concretely:

- argmax tie rule: reference takes the **first** index with `|x| == amax`; wasm/RISC-V take a global `min`/`max` comparison, so the sign of `iscale` (and hence every `qs` byte and `d`) can differ on any block with a symmetric `+v`/`-v` tie - normalised activations produce these routinely.
- out-of-range clamp: reference is `MIN(127, v)` plus a modulo-256 `int8_t` conversion (no lower clamp); wasm/RISC-V use saturating narrows.
- RISC-V additionally zeroes `bsums` in the all-zero case, where the reference leaves it untouched. A caller that reuses a `bsums` buffer across calls would see different values on RISC-V than on x86.

No repo code appears to depend on the untouched-`bsums` behaviour, so this is reported as a consistency observation, not as a bug.

### 4.2 Q2_K repack is gated on AVX-512

`ggml/src/ggml-cpu/repack.cpp:5039` selects `q2_K_8x8_q8_K` only when `ggml_cpu_has_avx512()`. There is no x86 `q2_K_8x8` body (only `arch/arm/repack.cpp` and the generic one), and the generic kernel offers no advantage over the non-repack path, so the gate is consistent with "do not enable generic repack" and the plan's phrasing of this as an AVX2-machine performance gap is not supported. Reported as an observation only.

### 4.3 An AVX-512 `quantize_row_q8_K` exists in-tree but is disabled and not bit-compatible

`ggml/src/ggml-cpu/amx/mmq.cpp:392` `quantize_row_q8_K_vnni`, disabled by `#if 1` at `:469` with the comment `// TODO: this is reference impl!`. It uses the opposite sign convention (`iscale = +127/amax`) and no `MIN(127, .)` clamp, so it cannot be enabled as-is without changing the numerics of every AMX matmul. Informational.

### 4.4 Upstream overlap

PR 22331 changes this same function on x86 (state `open`, created 2026-04-24). This experiment is a private fork experiment on branch `ai/simd-experiment`; nothing here is proposed upstream.
