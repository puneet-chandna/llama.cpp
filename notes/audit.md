# Phase 6: security and correctness audit

Scope: the one function this branch changes, `quantize_row_q8_K` in
`ggml/src/ggml-cpu/arch/x86/quants.c` (one hunk, +130/-3; the rest of the branch is
`notes/` and `tools/simd-experiment/`). The contract is byte-identical output to
`quantize_row_q8_K_ref`.

This audit is a written checklist with evidence. It is not a proof of absence of
bugs, and nothing in this file should be read as one.

## 1. Checklist

### 1.1 Out-of-bounds reads and writes: PASS

- Reads: the inner loops index `xb + 32*it + {0,8,16,24}` for `it < 8` (pass 1),
  `xb + 8*it` for `it < 32` (repair pass) and `xb + 64*j + {0..56}` for `j < 4`
  (pass 2). The largest offset is `64*3 + 56 + 7 = 255`, i.e. the last element of
  the block. `xb = x + i*QK_K` for `i < nb = k/QK_K`, so every access is inside
  `[0, nb*256) <= [0, k)`. Nothing reads the tail when `k % 256 != 0`.
- Writes: `qs` offsets are `64*j + {0, 32}` for `j < 4` (max `96+32-1 = 127` of a
  256-byte write region per `j`... precisely: two 32-byte stores per iteration at
  `64*j` and `64*j+32`, covering `[0, 256)` in total) and `bsums + 4*j` with an
  8-byte store (`[260, 292)` in the struct). Both stay inside the 292-byte block.
  `d` is the only other store.
- Evidence: ASan+UBSan over 3 560 000 blocks including the `odd_k` corpus
  (`k` in {0, 256, 640, 769, 2303}) and the `unaligned` corpus, no diagnostics.
  `simd-diff` also compares 32 sentinel bytes on each side of every output buffer,
  so a stray write would be reported as a mismatch.

### 1.2 Integer overflow / index arithmetic: PASS

`k` is `int64_t`, `nb = k / QK_K` is `int64_t`, `i` is `int64_t`, and the offset
`i*QK_K` is computed in `int64_t`. The loop counters `it` and `j` are `int` and
bounded by 8/32/4. `(int) imax` is safe because the value is below `0x7f800000` at
that point.

### 1.3 Misaligned loads and stores: PASS (with a note)

- Every vector access is `_mm256_loadu_ps`, `_mm256_storeu_si256` or
  `_mm_storel_epi64`; there is no aligned load/store in the function.
- Note: under UBSan's `-fsanitize=alignment` and a deliberately 1-byte-misaligned
  buffer, the *scalar* accesses `xb[best]` (line 573) and `yb->d` (line 628) plus
  the base-of-member expressions `yb->qs` (611, 612), `yb->bsums` (625) are
  reported. The reference reports the same class of diagnostics on more lines
  (`ggml-quants.c:2793,2795,2808,2809,2814,2816,2818`). No real caller can reach
  that state: the CPU backend's contract is 4-byte alignment (`params->wdata` is
  aligned and rows advance by a multiple of `sizeof(block_q8_K) = 292`, which is a
  multiple of 4), which satisfies the struct's own alignment requirement. The
  realistic case (4-byte aligned, not 32-byte aligned) produces no diagnostics at
  all and is what the sanitizer matrix run uses.
- The property that actually matters - never assuming `qs`/`bsums` are 32/16-byte
  aligned - is exercised by every run, because `qs` is at byte offset 4.

### 1.4 Uninitialized or partially written output: PASS

Every path writes `d`, `qs[0..255]` and `bsums[0..15]`:

- Fast path: `qs` is written by the two 32-byte stores per `j` (4 iterations cover
  all 256 bytes), `bsums` by the four 8-byte stores per `j` (16 int16 total), `d`
  last.
- Delegated paths (non-finite block, denormal/zero maximum, unreachable guards):
  the call to `quantize_row_q8_K_ref(xb, yb, QK_K)` writes the whole block with the
  reference's own semantics, including the case where the reference deliberately
  leaves `bsums` untouched (its `!amax` path). That is bit-identical by definition,
  and it is why delegating is safer than reimplementing those quirks.
- Accumulators (`imax0..3`) are initialized to zero before the loop; the horizontal
  reduction is complete (three steps over 8 lanes); `best` is checked before use.

### 1.5 Signedness and saturation: PASS

- `_mm256_maddubs_epi16(ones_u8, q)` has the unsigned constant as the first operand
  and the signed data as the second, which is the required order for multiplying
  `int8` data by `1`. Swapping them would corrupt every negative byte; the
  differential corpora (which include dense negative values) would fail.
- `bsums` bounds: byte pairs are within +/-254, 4-byte sums within +/-508, 16-byte
  sums within +/-2032; all fit `int16`/`int32` with no saturation.
- The missing `MIN(127, v)`: `cvtps2dq` rounds to nearest-even and the value is
  bounded by `127*(1+2^-23)`, so the result is within `[-127, 127]` and the
  `VPACKSSDW`/`VPACKSSWB` saturation - which would clamp exactly like `MIN(127, .)`
  - is not reached. See 1.6.

### 1.6 Floating-point rounding equivalence: PASS with a stated bound

- `nearest_int(v)` is `(v + 1.5*2^23)` then mantissa extraction, i.e. round-half-to-
  even for `|v| < 2^22`. `VCVTPS2DQ` is round-half-to-even per MXCSR. They agree for
  every product that occurs.
- The load-bearing bound: `iscale = -127/xmax` and `|x[j]| <= |xmax|`, with one
  rounding in the division and one in the multiply, gives
  `|iscale*x[j]| <= 127 * (1 + 2^-24)^2`, far below the 127.5 rounding threshold.
  `cvtps2dq` therefore returns at most 127, the reference's clamp `MIN(127, v)` is
  not reachable for finite products, and removing it is correct. (The comment in
  the source states this as "keeps every product below 127.5"; an earlier wording
  claimed `127 * (1 + 2^-23)` was itself in range, which was imprecise - the
  rounding is what brings it back into range. The independent review caught that.)
- Adversarial coverage for exactly this: the `clamp_edge` corpus constructs blocks
  where `iscale == 1.0f` and elements sit at `+-127.0`, `+-126.9999` and every exact
  half-integer, i.e. on the rounding and clamping boundaries. 10 000 000 blocks of
  it in the release sweep, 1 000 000 per compiler configuration, 200 000 under the
  sanitizers, 0 mismatches.
- Non-finite products are the case where this bound does not hold, and they are
  routed to the reference before any arithmetic happens (1.7).

### 1.7 NaN / infinity / denormal handling: PASS

- The maximum is taken on sign-cleared **integer** patterns, so `+infinity` and
  every NaN pattern compare above `0x7f800000`; one integer compare sends any block
  containing either to the reference. This is required: the reference's
  `nearest_int` on a non-finite product depends on the NaN payload (payload
  `0x400000` gives 0, `0x7fffff` gives 127). Without this guard the kernel produced
  `-128` where the reference produces `127` - the bug found by the fuzzer.
- Denormal and zero maxima: `iscale` overflows to infinity, handled by the
  `!isfinite(iscale)` guard (reference semantics: zeros for a denormal maximum, the
  `!amax` early-out - zero `qs`, untouched `bsums` - for an all-zero block).
- Argmax tie rule: the reference keeps the **first** element with `|x| == amax`
  (strict `>`), and its sign determines `iscale`. The repair pass uses
  `cmpeq_epi32` on cleared sign bits plus `movemask` + `ctz`, i.e. the lowest lane
  of the earliest vector = the first element in memory order, and stops at the first
  match. Covered by the `tie` corpus.
- `-0.0` cannot be confused with `+0.0`: the sign bit is cleared before any compare,
  and 0x00000000 != 0x80000000 only differs in that very bit.

### 1.8 `k` not a multiple of 256, `k == 0`, `nb == 0`: PASS

`nb = k / QK_K` (truncating) drives the loop, so a tail shorter than a block is
neither read nor written, exactly like the reference; `k == 0` makes the loop body
execute zero times. The caller contract is `k % 256 == 0` (asserted here and in the
reference). Covered by the `odd_k` corpus under ASan.

### 1.9 Other undefined behaviour: PASS

No shifts on data (only `__builtin_ctz` on a nonzero mask, guarded by `m != 0`), no
float-to-int cast (all conversions are `VCVTPS2DQ`, whose out-of-range behaviour is
a defined sentinel rather than C UB, and the guard excludes the inputs that would
reach it), no pointer punning beyond `_mm256_castps_si256` / `_mm256_castsi256_ps`
(intrinsic-level, defined), no aliasing violation (`x` is `const float *`,
`vy`/`y` is `block_q8_K *`, and their relationship is fixed by the callers). The
`__m256i`-typed stores are performed through `_mm256_storeu_si256`, which is the
intrinsic for exactly that purpose.

### 1.10 Reachability without AVX2: PASS

The body is inside `#if defined(__AVX2__)`, the `#else` branch calls the reference,
and there is no runtime dispatch in this function, so no CPU without AVX2 can
execute these instructions. Verified by building and running the `minfeat`
configuration (AVX2 only) and by the opcode census in `notes/verification.md`
section 3: no FMA / AVX-VNNI / F16C / AVX-512 instruction is present even when the
compiler is allowed to emit all of them.

## 2. Static analysis

| tool | scope | findings in the changed file | findings elsewhere (pre-existing) |
|---|---|---|---|
| clang-tidy 21.1.8 (`-*,bugprone-*,clang-analyzer-*,cert-*,performance-*`, `-header-filter=.*`) | `arch/x86/quants.c` | **0** | 14 in `ggml/src/ggml-cpu/simd-mappings.h` (`bugprone-macro-parentheses` in existing macros), 68 818 suppressed in non-user code |
| scan-build 21.1.8 | `ggml-cpu` target, file force-rebuilt | `No bugs found` (exit 0) | - |
| cppcheck 2.22.0 (`--enable=warning,style,performance,portability`, real compile flags from `compile_commands.json`) | `arch/x86/quants.c` | **0** | 8 in `ggml/src/ggml-impl.h` (null-pointer redundancy, `const` hints, two `uninitvar` in existing helpers) |

No finding was classified as a false positive in the changed file because there were
none in it.

## 3. Independent fresh-context review

A fresh-context review was run by a separate agent (different model, no access to
these notes, no conversation history). It was given only: the diff
(`git diff upstream/master...HEAD -- ggml/src/ggml-cpu/arch/x86/quants.c`), the
reference implementation and `nearest_int`, the `block_q8_K` layout, the callers
that define the `k % 256 == 0` contract, and the same 10-item checklist as section 1.
It ran nothing and read no test results.

**Verdict: items 1-10 all PASS, zero critical/high/medium findings, two
informational notes.**

Its independent confirmations, which add to (not replace) section 1:

- The largest element index touched is `255` of the block in every loop, and the
  stores cover `qs[0..255]` and `bsums[0..15]` exactly.
- `(int) imax` (line 556) is provably safe: it runs only after the
  `imax >= 0x7f800000u` guard, so `imax <= 0x7f7fffff < INT32_MAX` and
  `_mm256_set1_epi32` sees the same bits the `andnot` produced.
- `nearest_int` is round-half-to-even for `|fval| <= 4194303`, `cvtps2dq` is the same
  function in the default MXCSR mode, `iscale` is computed from the same scalar
  expression on the same element, and `iscale*x[j]` is a single-rounded multiply in
  both.
- `bsums` bounds as it recomputed them: byte pairs at most +/-254, 4-byte sums
  +/-508, 16-byte sums +/-2032, all losslessly narrowed by the final `packs`.
- The tie rule, the NaN/Inf/denormal delegation boundary, the `!amax` delegation
  (byte-identical, because the delegate does the same thing to the same block) and
  the alignment discipline are all reproducible from the code as written.
- `GGML_RESTRICT` on `x`/`vy` is not violated by the delegate call; no caller
  aliases the two buffers.
- The `#if defined(__AVX2__)` guard exactly matches the instruction set actually
  used (no AVX-512, no FMA), and there is no runtime dispatch in this file.

Its two informational findings, both **acted on or accepted**:

1. **Comment arithmetic (acted on).** The justification comment claimed
   `|iscale*x| <= 127 * (1 + 2^-23)`, which does not by itself prove "already in
   range" (the value is marginally above 127, it is the *rounding* that brings it
   back). The reviewer verified that the true worst case is
   `127 * (1 + 2^-24)^2`, far below the 127.5 rounding threshold, so the code is
   correct, but asked for a self-contained statement. The comment now reads
   "`|x| <= |max|` keeps every product below 127.5, so cvtps2dq returns at most
   127". This is a comment-only change, and it was verified to be codegen-neutral:
   all six build configurations rebuilt with identical opcode fingerprints and 0
   mismatches on a 3 200 000-block sweep each, and the fuzz library still replays
   the archived crash input cleanly (`build/logs/phase6_rebuild_check.txt`).
2. **MXCSR dependency (accepted, documented).** `cvtps2dq` uses the ambient rounding
   mode; `nearest_int` does not. llama.cpp never changes MXCSR, the pre-existing
   AVX2 `quantize_row_q8_0`/`q8_1` quantizers make the same assumption, and the
   comment already says "from MXCSR". No action.

It also noted that `__builtin_ctz` is a GCC/Clang builtin (consistent with the rest
of the file's GNU-isms); the `m != 0` guard makes its semantics unambiguous. No
action.

The reviewer's own scope statement: it did not check anything requiring execution
(no fuzzing, no sanitizers, no differential runs) - that evidence is section 1 of
this file and `notes/verification.md`.

## 4. What could not be checked

- No runtime ISA dispatch exists in this function, so "wrong instruction on an
  older CPU" is a compile-time property only; it was verified on the one CPU
  available (Alder Lake). Nothing here proves the AVX2 body is optimal for other
  microarchitectures, only that it is correct and faster on this one.
- The sanitizer runs use ASan+UBSan, not MSan (the tree's build options for
  MemorySanitizer are not wired for this target), so uninitialized *reads* of
  library-internal buffers would not be caught by a sanitizer; they are covered
  instead by the sentinel-byte comparison (a read of uninitialized memory would
  change the output and show up as a mismatch) and by the `--self-test`
  mechanism checking that the comparison itself works.
- Valgrind and `perf` were not usable (no `sudo`, `perf_event_paranoid = 4`).
- No claim of formal verification: the equivalence argument rests on the checked
  bounds in section 1.6 plus the differential and fuzz evidence, not on a proof
  assistant.
