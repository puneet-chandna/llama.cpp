# REPORT: an AVX2 `quantize_row_q8_K` for x86

Branch `ai/simd-experiment`, based on `upstream/master` = `b5cb8f6f4`. The
production change is one function in one file
(`ggml/src/ggml-cpu/arch/x86/quants.c`, one hunk, +130/-3).

## 1. What was done

`quantize_row_q8_K` on x86 was a forwarder to the portable C reference
(`ggml/src/ggml-quants.c:2784`). It now has an AVX2 implementation that is
**byte-identical** to that reference and 10.3-10.6x faster on the activation row
lengths these models use. Q8_K is the activation type of every K-quant and
IQ-quant matmul, so this function runs once per activation row per quantized
weight tensor.

Why this kernel and not another: `notes/research.md` sections 1-3 audit every
generic fallback reachable on this CPU (Alder Lake, AVX2/FMA/F16C/AVX-VNNI, no
AVX-512) and show that this is the only reachable one that is hot at the same time.
`notes/hotness.py` and the `rdtsc` measurement in `notes/baseline.md` section 6
quantify the share (0.86-6.44% of prefill wall time depending on the model and on
whether weight repacking is enabled).

## 2. Method

- **Correctness oracle**: the existing reference function, with **zero tolerance**
  (any differing byte fails). Every output byte is discrete, so this is an equality
  test. `notes/baseline.md` section 1 explains why zero is achievable here and why
  the same rule is *not* available to the pre-existing `q8_0` pair.
- **Tools added** (not part of the library, new directory `tools/simd-experiment/`):
  `simd-bench` (paired reference/candidate timing in one process),
  `simd-diff` (16 adversarial corpora, byte-exact comparison of the full 292-byte
  block plus sentinels, with `--self-test`), `fuzz-q8k` (libFuzzer + ASan harness
  that runs both implementations on random `k`/offset/payload combinations).
- **Measurement configuration** for every end-to-end number: `taskset -c 0,2`
  (one hardware thread per P core, no SMT sharing), `-t 2`, same flags, same
  repetitions on both sides. The "before" binary is a `git worktree` of this
  branch's base commit; `objdump` shows its `quantize_row_q8_K` is
  `jmp quantize_row_q8_K_ref@plt`.
- **Drift control**: this host drifts by 2-6% between runs minutes apart (measured,
  `notes/baseline.md` section 3), which is the same order as the expected effect, so
  the headline end-to-end comparison is an **interleaved A/B** run that alternates
  the two binaries one repetition at a time on the same core pair.

## 3. Results: the kernel

`simd-bench --impl both --reps 41`, GCC 15.2.0 `-O2 -march=native`, pairs measured
in the same process and reported per repetition:

| k | reference ns | candidate ns | paired ratio | ratio sd | candidate ns/element |
|---|---|---|---|---|---|
| 256 | 546 | 67 | 8.14x | 0.38 | 0.262 |
| 512 | 1093 | 121 | 9.01x | 0.20 | 0.236 |
| 1024 | 2167 | 222 | 9.77x | 0.88 | 0.217 |
| **2048** | 3755 | 365 | **10.29x** | 1.09 | 0.178 |
| **3072** | 5478 | 517 | **10.59x** | 0.92 | 0.168 |
| 4096 | 7077 | 723 | 9.81x | 1.12 | 0.177 |
| 8192 | 14598 | 1466 | 10.08x | 0.45 | 0.179 |
| 32768 | 58127 | 6074 | 9.52x | 0.12 | 0.185 |

`k = 2048` and `k = 3072` are the activation row lengths of the models used end to
end (1B/7B and 3B respectively). The reference costs about 2.11 ns per element and
the candidate about 0.17-0.26 ns, i.e. roughly one cycle per element; the saving is
about 1.85 ns per quantized element.

## 4. Results: correctness

| what | coverage | result |
|---|---|---|
| differential, all corpora | **252 760 000 blocks = 6.47e10 elements** across 6 build configurations (release sweep, 4 compiler/flag builds, ASan+UBSan) | 0 mismatches |
| differential, pre-fix kernel (history) | 238 560 020 blocks = 6.11e10 elements | 0 mismatches |
| libFuzzer + ASan, 900 s | 16 007 485 executions (17 766 exec/s), 43/148 edges/features covered | 0 failures |
| `test-quantize-fns` (ASan+UBSan) | - | exit 0 |
| `test-backend-ops -o MUL_MAT` | 2794 tests | 2794/2794 passed |
| `ctest` (ASan+UBSan), 67 tests | full suite | 61/67, the 6 failures are byte-identical to the same suite on the pristine tree |
| `simd-bench --check` | all `k` | exact |
| `objdump` fingerprint | 6 build configurations | no FMA/VNNI/F16C/AVX-512 opcode; integer max present; no float max, no clamp |
| static analysis | clang-tidy 21.1.8, scan-build 21.1.8, cppcheck 2.22.0 | 0 findings in the changed file |

The corpora include NaN (many payloads), infinities, denormals, signed zeros,
`+-FLT_MAX`, exact half-integer quantization boundaries, the argmax tie rule,
`k % 256 != 0`, and 1..31-byte misalignments. Details and the raw logs are in
`notes/verification.md`.

## 5. The bug the fuzzer found (and the fix)

The differential corpora were not enough. A block holding six NaNs **together with
a finite maximum** produced `-128` where the reference produces `127`: the
reference's `nearest_int` on a non-finite product is a function of the NaN
**payload**, so its result is not bounded by 127, and my "the clamp is unreachable"
argument had silently assumed finite products. All three earlier variants (A, B, C)
had it. libFuzzer found it in 53 executions.

The fix takes the block maximum in the **integer domain** (cleared sign bits, so
integer order equals float order for non-negative values and `VPMAXSD` has no NaN
special case), which also detects NaN and infinity for free: every NaN pattern
compares above `0x7f800000`. Any block containing a non-finite value is handed to
the reference. The fix made the kernel *faster* (8.89x -> 10.29x at `k = 2048`).

Regression protection: a new `nan_mix` corpus (fails 200/200 on the pre-fix kernel,
passes on the fixed one) and the archived crashing input
(`notes/fuzz-seeds/fuzz-q8k-nan-finite-mix.bin`, copied into the fuzz corpus by
`notes/run_fuzz.sh`). Full write-up: `notes/impl.md`, "Variant D".

## 6. End-to-end (llama-bench / llama-perplexity)

### 6.1 Perplexity: bit-identical, as bit-exactness predicts

`llama-perplexity -c 512 -t 2`, pinned to cores 0 and 2, same file, same flags,
same chunks. The "before" side was measured 5 times (5 for 1B, 5 for 3B, 3 for 7B)
and produced the same value every time, so run-to-run variation is exactly zero on
this host; the "after" side was measured 2/2/1 times.

| model | chunks | before (per rep) | after (per rep) | delta |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct Q4_K_M | 16 | 16.4452 | 16.4452 | 0 |
| Llama-3.2-3B-Instruct IQ4_XS | 16 | 12.4257 | 12.4257 | 0 |
| Qwen2.5-7B-Instruct Q4_K_M | 8 | 6.7962 | 6.7962 | 0 |

Every printed digit matches, on ten before-reps and five after-reps. This is the
acceptance criterion met in its strongest form: because the kernel is bit-exact
(section 4) the quantized activations are bit-identical, therefore the forward pass
is bit-identical, therefore the perplexity cannot move. A non-zero delta would have
been a bug report.

Logs: `build/logs/ppl_before-pristine_*_*.log` and
`build/logs/ppl_after-fixed_*_*.log`, summary in
`build/logs/ppl_after_summary.txt`.

### 6.2 Throughput: below this machine's resolution

The only valid throughput comparison on this host is the **paired interleaved** run
(`notes/run_ab_interleaved.sh`): the two binaries alternate one `llama-bench`
repetition at a time on the same pinned core pair, so drift affects both sides and
cancels in the per-repetition ratio. Cross-session comparison is not valid here - the
*same* "before" library measured 71.28 t/s on 1B pp512 in the baseline session and
89.56 t/s in this one (25% apart), and the after-side warm medians in
`build/logs/bench_after-fixed.txt` and `build/logs/bench_nr_after-fixed.txt` come out
11-39% above the baseline table for *every* model, including ones whose share is
under 1%. That is machine state, not the kernel.

Mean per-repetition ratios (after/before), 5 paired repetitions per cell, both
repack settings, against the prediction from measured cycle share x kernel speedup
(`notes/baseline.md` section 6):

| model / test | repack | before t/s | after t/s | mean paired delta | se | predicted |
|---|---|---|---|---|---|---|
| 1B pp512 | 1 (default) | 89.56 | 90.79 | -1.10% | 1.75% | +1.25% |
| 1B tg128 | 1 | 25.02 | 25.25 | +2.65% | 2.61% | +0.28% |
| 3B pp512 | 1 | 30.63 | 31.14 | +0.44% | 1.14% | +4.35% |
| 3B tg128 | 1 | 9.41 | 9.06 | -2.10% | 1.57% | +1.04% |
| 7B pp256 | 1 | 13.48 | 13.42 | -1.15% | 0.45% | +0.77% |
| 7B tg64 | 1 | 4.46 | 4.43 | -0.92% | 1.38% | +0.23% |
| 1B pp512 | 0 | 94.06 | 92.91 | -0.44% | 1.08% | +5.81% |
| 1B tg128 | 0 | 23.62 | 24.09 | +0.82% | 1.50% | +1.49% |
| 3B pp512 | 0 | 29.66 | 30.42 | +1.91% | 1.05% | +4.45% |
| 3B tg128 | 0 | 8.44 | 8.65 | +0.99% | 3.41% | +1.04% |
| 7B pp256 | 0 | 14.90 | 15.10 | +1.86% | 0.81% | +3.18% |
| 7B tg64 | 0 | 4.78 | 4.86 | -0.04% | 3.80% | +0.27% |

The before/after columns are the medians of the five repetitions on each side. Raw
data: `build/logs/ab_interleaved.csv` (default repack) and
`build/logs/ab_interleaved_rpk0.csv`.

**This method cannot resolve effects of this size on this host, and the data proves
it without appeal to theory.** The 3B model is IQ4_XS, which has no repackable
tensor, so its `--repack 0` and `--repack 1` runs perform the *same work through the
same kernel*. Compared against each other, that identical workload came out at
-2.10% and +0.99% on tg128 (5 pairs each): a 3.1 point swing with no cause. The real
resolution of the whole table is therefore about +/-3%, every observed delta between
-2.1% and +2.7% lies inside it, and nothing here is a confirmed gain or a confirmed
regression.

Two things can still be said:

- **The predictions are not reproduced at the top end.** The four largest predicted
  cells (+5.81%, +4.45%, +4.35%, +3.18%) measured -0.44%, +1.91%, +0.44%, +1.86%.
  That shortfall is 1-6 standard errors, and I could not identify its cause with the
  tools available: `perf` and RAPL are blocked on this host (`perf_event_paranoid=4`,
  no `sudo`), so the frequency and power state during each run are unknown. On a
  15 W part, part of a saved cycle budget can come back as a lower sustained clock,
  which would look exactly like this - but that is a hypothesis, not a measurement.
- **The kernel-level result does not depend on any of this.** 10.29x at `k=2048` and
  10.59x at `k=3072`, measured in a single binary against the reference, with 6.47e10
  elements verified bit-identical. The defensible end-to-end statements are the
  perplexity one (section 6.1: exact equality on all three models) and "the
  throughput effect is below this machine's ~3% resolution".

## 7. Observations about pre-existing code (not fixed here)

Reported because they were found while building the evidence, not because they are
in scope for this change:

1. **The existing AVX2 `quantize_row_q8_0` is not bit-exact against
   `quantize_row_q8_0_ref`.** Measured with the same harness:
   85 453 differing blocks out of 1 600 000 across the corpora, concentrated in
   `inf`, `nan`, `mixed_nonfinite`, `nan_mix`, `clamp_edge`, `denormal`, `odd_k`,
   `random_mix` (0 for `uniform`, `wide`, `zero`, `allzero`, `extreme`). The causes
   are visible in the source: the reference computes `id = 1.0f/(amax/127)` while
   the AVX2 path computes `127.f/amax` (one rounding instead of two, so the
   products differ by about 1 ulp and exact half-integer cases round differently),
   the reference rounds with `roundf` (half away from zero) while the AVX2 path
   uses `_MM_ROUND_NEAREST` (half to even), the reference's `MAX` macro ignores NaN
   while `_mm256_max_ps` propagates it, and for an out-of-range float the reference
   truncates a `double` to `int8_t` (implementation-defined, observed `0x00`) where
   `VPACKSSWB` saturates (`0x80`). This does not affect my change, and I did not
   touch it.
2. **`arch/x86/quants.c` cannot be compiled with FMA or F16C disabled**, for
   pre-existing reasons: unguarded `_mm256_fmadd_ps` in the vector dot product
   helpers (lines 728 and 730), and `arch/x86/repack.cpp` passing
   `const ggml_half *` where the non-F16C helper wants `ggml_fp16_t *`. This is why
   the minimum-feature-flag proof for my kernel is the `minfeat` build (AVX2 on,
   AVX-512/AVX-VNNI off, no `-march=native`) plus the opcode census, rather than a
   `-mno-fma` build of the file.
3. **The wasm and RISC-V q8_K quantizers are not bit-compatible** with the
   reference (different sign rule, truncation instead of round-to-nearest-even).
   Recorded in `notes/research.md` section 3; it is why the reference is the only
   acceptable oracle here.
4. **The Q2_K repack kernels are gated on AVX-512** even though an AVX2 body
   exists nearby (`arch-fallback.h` x86 branch), so they are dead code on this CPU.
   Observation only.
5. **Upstream PR 22331** ("ggml-cpu: optimize q8 quantization on x86 SIMD", open
   since 2026-04-24) targets this same function. This branch is a private fork
   experiment: nothing here is proposed upstream and that PR's diff was not used as
   a source.

## 8. Limitations and what was not checked

- Absolute numbers are specific to this host (i5-1235U, 12 threads, `powersave`
  governor, no turbo control) and drift by 2-6% between sessions. Ratios measured
  in-process or interleaved are the trustworthy part.
- `perf` is unusable (`perf_event_paranoid = 4`, no `sudo`), so the share of time
  was measured with `rdtsc` instrumentation of the baseline build instead
  (`notes/instrumentation.diff`, reverted; the measured library was rebuilt and its
  `sha256` matches the uninstrumented one).
- The 7B model runs memory-limited (`MemAvailable` was 2.6-3.5 GB against a 4.7 GB
  model), so its numbers carry extra noise and are marked as such; 1B and 3B carry
  the conclusions.
- Single-variant build, no runtime ISA dispatch: the AVX2 body is compile-time
  guarded, and non-AVX2 x86 gets the reference.
- I did not verify the q8_K semantics against a *mathematical* definition of
  quantization, nor against other backends (CUDA/Metal/Vulkan); the claim is
  equivalence to this repository's own reference implementation, which is the
  contract the function has.
- The static analysis tools report nothing in the changed file, but static analysis
  cannot prove absence of bugs; the differential and fuzz evidence carry that
  weight.
- Nothing was submitted upstream, no PR was opened, and no other branch was
  touched.

## 9. Artifacts

| file | what it is |
|---|---|
| `ggml/src/ggml-cpu/arch/x86/quants.c` | the change (one function) |
| `tools/simd-experiment/{simd-bench,simd-diff,fuzz-q8k}.cpp`, `corpus.h` | the harnesses used for every claim |
| `notes/REVIEW.md` | line-by-line walkthrough of the change |
| `notes/impl.md` | implementation history, variants A-D, the fuzzer bug and fix |
| `notes/verification.md` | all verification runs and their raw numbers |
| `notes/baseline.md` | tolerance rule, measurement config, before-side numbers, share |
| `notes/research.md` | candidate audit and why this kernel was chosen |
| `notes/audit.md` | security/correctness checklist, findings, static analysis |
| `notes/env.md` | host, toolchain, models, build commands |
| `notes/fuzz-seeds/` | the archived crashing input |
| `notes/instrumentation.diff` | the `rdtsc` patch used for the share measurement (not applied) |

## 10. How to reproduce

Toolchain: CMake 4.4.4 (`pip install --user cmake`), GCC 15.2.0, Clang 21.1.8 in
`~/opt/clang-21`, libFuzzer from that Clang.

```sh
# primary measurement build
~/.local/bin/cmake -B build/release -S . -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
~/.local/bin/cmake --build build/release -j 8 --target llama-bench llama-perplexity simd-bench simd-diff

# sanitizer build (differential matrix + ctest)
~/.local/bin/cmake -B build/asan -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_NATIVE=ON \
    -DGGML_SANITIZE_ADDRESS=ON -DGGML_SANITIZE_UNDEFINED=ON -DLLAMA_SANITIZE_ADDRESS=ON \
    -DLLAMA_SANITIZE_UNDEFINED=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
~/.local/bin/cmake --build build/asan -j 8
```

```sh
# correctness and kernel speed
build/release/bin/simd-diff --self-test
build/release/bin/simd-diff --corpus all --blocks 10000000 --seed 1
build/release/bin/simd-bench --impl both --k 2048,3072 --reps 41 --check
build/release/bin/test-backend-ops test -b CPU -o MUL_MAT

# fuzzing (900 s) and the compiler/flag matrix
notes/run_fuzz.sh 900
for d in gcc-o2 clang-o2 clang-o3 minfeat; do build/$d/bin/simd-diff --corpus all --blocks 1000000; done

# end to end, pinned to one thread per P core
notes/run_bench.sh          build/logs/bench_after-fixed.txt    after-fixed
notes/run_bench_norepack.sh build/logs/bench_nr_after-fixed.txt after-fixed
notes/run_ab_interleaved.sh <before-bin-dir> build/release/bin  build/logs/ab_interleaved.csv 5
```

`simd-diff` takes `--corpus --blocks --calls --k --seed --impl --misalign
--self-test --q8-0-baseline --emit-corpus --quiet`; `simd-bench` takes `--impl --k
--mode --rows --reps --seed --check --csv`. Both print their usage with `--help`.
