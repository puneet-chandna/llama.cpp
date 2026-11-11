# Phase 3: tooling, bit-exactness rule and baseline

Artifacts in this phase:

| file | content |
|---|---|
| `tools/simd-experiment/{CMakeLists.txt,corpus.h,simd-bench.cpp,simd-diff.cpp,fuzz-q8k.cpp}` | microbenchmark, differential tester, libFuzzer harness |
| `notes/baseline_raw.txt` | full "before" output: llama-bench (default) + llama-perplexity for all three models |
| `notes/baseline_raw_norepack.txt` | full "before" output: llama-bench with weight repacking disabled |
| `notes/hotness.py` | per model tensor inventory: elements quantized per token and per prefill batch |
| `notes/run_bench.sh`, `notes/run_bench_norepack.sh` | the measurement drivers, used unchanged for the phase 5 re-run |
| `notes/instrumentation.diff` | temporary rdtsc patch used only to measure the share of time |

## 1. Tolerance rule: zero bytes

**Any byte difference between `quantize_row_q8_K` and `quantize_row_q8_K_ref` is a
failure.** `simd-diff` compares the whole 292-byte `block_q8_K` (the 32-bit `d`,
the 256 `qs` bytes and the 16 `bsums` int16 values) plus sentinel bytes before and
after the output region, so an out-of-bounds write is a failure too. There is no
tolerance parameter and none may be added later.

Justification, in order:

1. Every output is discrete. `qs` and `bsums` are integers and `d` is a 32-bit
   float computed by a scalar expression that the kernel replicates literally
   (`-127.f/max`, then `1/iscale`); there is no fp16 rounding step in this path
   (see `notes/research.md` 4.6).
2. The reference's rounding is `nearest_int` = round-half-to-**even** (magic add
   of `1.5 * 2^23`), and `VCVTPS2DQ` is round-half-to-even from MXCSR, so the two
   agree for every finite product - they do not merely agree "usually".
3. Therefore bit-exactness is achievable, and the approved plan requires it: a
   bit-identical quantizer makes the whole forward pass bit-identical, which turns
   the end-to-end check into an equality test instead of a tolerance test.

The empirical shortcut the plan suggested for point 3 is **not** available: the
existing AVX2 `quantize_row_q8_0` is not bit-exact against its reference
(measured: 14510 differing blocks in 300000 across the corpora; cause and
evidence in `notes/research.md` 4.5). The rule therefore rests on the analytical
argument above and is verified by measurement rather than inherited.

## 2. Measurement configuration

| parameter | value |
|---|---|
| pinning | `taskset -c 0,2` (one hardware thread per P core, no SMT sharing) |
| threads | `-t 2` |
| llama-bench reps | 5 (3 for the 7B, which is memory limited on this host) |
| llama-perplexity | `-c 512 --chunks 16 -t 2`, 5 repetitions (8 chunks, 3 repetitions for the 7B) |
| "before" binary | `/home/puneet/llama-pristine/build/release/bin`, git `5fac66808` (phase 2 commit), `sha256(libggml-cpu.so)` = `17e8c6a9d07633d30182c3a1f76339e7` |
| verifier | `objdump` shows `quantize_row_q8_K` = `jmp quantize_row_q8_K_ref@plt` in that library |

The "before" build is a `git worktree` of the phase 2 commit, so the working tree
under test was never rebuilt with the reference implementation by hand.

Two notes on the drivers:

- `llama-bench` does **not** accept the common `-nr` / `--no-repack` flag
  (`common/arg.cpp:2395` belongs to the `gpt_params` parser). It has its own
  `--repack <0|1>` option (`tools/llama-bench/llama-bench.cpp:915`), which is what
  `notes/run_bench_norepack.sh` uses. The first baseline attempt used `-nr`; all
  three no-repack llama-bench runs failed with a usage error, and the default-config
  numbers from that attempt are valid. The gaps were filled by
  `notes/run_bench_norepack.sh`.
- A single driver run covering all three models with both perplexity and both
  repack settings exceeds the 3600 s tool cap, so the default and no-repack
  measurements are two files, each produced in one uninterrupted run per side.

## 3. Baseline: llama-bench (before)

Default configuration (repacking enabled):

| model | test | t/s |
|---|---|---|
| Llama-3.2-1B-Instruct Q4_K_M | pp512 | 71.28 ± 3.89 |
| Llama-3.2-1B-Instruct Q4_K_M | tg128 | 17.50 ± 0.93 |
| Llama-3.2-3B-Instruct IQ4_XS | pp512 | 25.39 ± 0.74 |
| Llama-3.2-3B-Instruct IQ4_XS | tg128 | 7.30 ± 0.21 |
| Qwen2.5-7B-Instruct Q4_K_M | pp256 | 12.20 ± 0.50 |
| Qwen2.5-7B-Instruct Q4_K_M | tg64 | 3.87 ± 0.23 |

Weight repacking disabled (`--repack 0`), so every K/IQ matmul uses the candidate:

| model | test | t/s |
|---|---|---|
| Llama-3.2-1B-Instruct Q4_K_M | pp512 | 83.05 ± 3.51 |
| Llama-3.2-1B-Instruct Q4_K_M | tg128 | 23.68 ± 0.84 |
| Llama-3.2-3B-Instruct IQ4_XS | pp512 | 26.85 ± 0.28 |
| Llama-3.2-3B-Instruct IQ4_XS | tg128 | 7.65 ± 0.44 |
| Qwen2.5-7B-Instruct Q4_K_M | pp256 | 12.99 ± 0.66 |
| Qwen2.5-7B-Instruct Q4_K_M | tg64 | 4.36 ± 0.06 |

The `±` is the run-to-run standard deviation over the repetitions, as reported by
llama-bench.

Two observations worth recording because they bound what an end-to-end result can
mean:

- The IQ4_XS model has **no** repackable tensor on this CPU (`notes/hotness.py`
  reports 0 of 197), yet its pp512 differs by 5.8% between the two runs above
  (25.39 vs 26.85). The runs were about 45 minutes apart with different memory
  pressure (8.5 GB vs 11.4 GB `MemAvailable`). Cross-run comparisons on this host
  therefore carry a few percent of drift, which is the same order as the effect
  this change can produce. Phase 5 adds an interleaved before/after measurement to
  control for it.
- Repacking makes pp *slower* for the 1B on this host (71.28 vs 83.05 t/s). This is
  not investigated further here - it is a property of the repack gemm path versus
  the plain vec_dot path on this model shape, and it is not affected by this
  change - but it means "no-repack" is not a slower configuration to be used as a
  worst case.

## 4. Baseline: llama-perplexity (before)

| model | chunks | reps | PPL per rep | reported `±` |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct Q4_K_M | 16 | 5 | 16.4452, 16.4452, 16.4452, 16.4452, 16.4452 | 0.74551 |
| Llama-3.2-3B-Instruct IQ4_XS | 16 | 5 | 12.4257, 12.4257, 12.4257, 12.4257, 12.4257 | 0.54566 |
| Qwen2.5-7B-Instruct Q4_K_M | 8 | 3 | 6.7962, 6.7962, 6.7962 | 0.39313 |

**llama-perplexity is deterministic on this host**: every repetition of the same
configuration produced a bit-identical estimate, including the 7B whose third
repetition was measured in a separate process half an hour later. The `±` value
that llama-perplexity prints is the standard error across chunks, *not* run-to-run
variation; the run-to-run variation is exactly zero.

Consequence for the acceptance criterion: the kernel is bit-exact, therefore the
quantized activations are bit-identical, therefore the model output must be
bit-identical, therefore **the after-change PPL must be exactly equal** to the
values above. Any non-zero delta is a bug report, not noise. This is a stronger
statement than the plan's "delta within 3 sigma of the baseline repetitions",
which is satisfied trivially once PPL is bit-identical.

## 5. Kernel baseline and predicted share of time

`quantize_row_q8_K_ref` cost, `build/release/bin/simd-bench --impl ref`, one call
of `k` elements per sample, 15-25 samples, pinned to cores 0 and 2:

| k | ns per element |
|---|---|
| 256 | 2.13 |
| 512 | 2.11 |
| 2048 | 2.12 |
| 4096 | 2.11 |
| 8192 | 2.25 |
| 32768 | 2.32 |

(The large-`k` values are noisier because the working set no longer fits as well in
L1/L2; the small-`k` values are the relevant ones since real activation rows on
these models are 2048 and 3072 elements.)

Elements quantized per token, from `notes/hotness.py` (sum of `ne[0]` over every
weight tensor whose `vec_dot_type` is `Q8_K`, plus the output head):

| model | default config | with repacking disabled |
|---|---|---|
| Llama-3.2-1B-Instruct Q4_K_M | 83 968 (328 blocks) | 329 728 (1288 blocks) |
| Llama-3.2-3B-Instruct IQ4_XS | 748 544 (2924 blocks) | 748 544 (identical: nothing is repackable) |
| Qwen2.5-7B-Instruct Q4_K_M | 322 560 (1260 blocks) | 1 139 712 (4452 blocks) |

Predicted share of wall time = elements per token x 2.11 ns x measured t/s, and the
predicted end-to-end gain assumes the kernel becomes free (an upper bound):

| model / config | pp share | pp gain upper bound | tg share | tg gain upper bound |
|---|---|---|---|---|
| 1B default | 1.26% | 1.26% | 0.31% | 0.31% |
| 1B no-repack | 5.78% | 5.78% | 1.65% | 1.65% |
| 3B (both configs) | 4.01% | 4.01% | 1.15% | 1.15% |
| 7B default | 0.83% | 0.83% | 0.26% | 0.26% |
| 7B no-repack | 3.12% | 3.12% | 0.30% | 0.30% |

These are the numbers the phase 5 "too good" check uses: an observed pp gain above
twice the corresponding share cannot be real and must be investigated.

## 6. Share measured with rdtsc

`perf` is unusable on this host (`perf_event_paranoid = 4`, no sudo), so the share
was measured by instrumenting the *baseline* build (the disposable worktree) with
`rdtsc` around the call to `quantize_row_q8_K_ref`. The patch is
`notes/instrumentation.diff`; it adds a constructor-free counter set and a
destructor that prints `Q8K_INSTR cycles=... calls=... elems=...` to stderr, and it
was reverted before anything was committed. The measured library was rebuilt from
the pristine source afterwards and its `sha256` is byte-identical to the one used
for the baseline benchmarks (`17e8c6a9d07633d30182c3a1f76339e7`), and `objdump`
shows the `jmp quantize_row_q8_K_ref@plt` forwarder again, so the "before" binary
used later is the uninstrumented one.

Method: `llama-bench -p <pp> -n 0 -t 2 -r 3 --no-warmup --repack <0|1>`, pinned to
cores 0 and 2. `--no-warmup` matters: llama-bench otherwise runs one unmeasured
warmup pass, and the counters cannot tell warmup from timed passes. Nothing else
was running on the machine. TSC frequency measured with a one-second `rdtsc`
calibration: 2.49597 GHz.

Share = (cycles in the quantizer per pass) / (TSC ticks in one pass), where one
pass is `pp` tokens at the t/s that llama-bench reports for the same run.

| model | repack | pp t/s | pp seconds | cycles/pass in quantizer | elements/pass | elements/token | share of pp wall time |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | on | 77.85 ± 1.38 | 6.577 | 227 126 267 | 37 763 072 | 73 756 | **1.384%** |
| Llama-3.2-1B Q4_K_M | off | 83.11 ± 0.89 | 6.161 | 989 758 893 | 161 495 040 | 315 420 | **6.437%** |
| Llama-3.2-3B IQ4_XS | on | 26.70 ± 0.67 | 19.176 | 2 298 403 740 | 374 359 040 | 731 170 | **4.802%** |
| Llama-3.2-3B IQ4_XS | off | 27.29 ± 0.97 | 18.762 | 2 300 183 361 | 374 359 040 | 731 170 | **4.912%** |
| Qwen2.5-7B Q4_K_M | on | 11.76 ± 0.55 | 21.769 | 465 133 627 | 75 920 384 | 296 564 | **0.856%** |
| Qwen2.5-7B Q4_K_M | off | 12.88 ± 0.44 | 19.876 | 1 747 148 057 | 283 276 288 | 1 106 548 | **3.522%** |

Cross-checks on this table:

- The two IQ4_XS rows have **different t/s but identical element counts and cycle
  counts** (374 359 040 elements, 2.298 vs 2.300 Gcycles per pass). IQ4_XS has no
  repackable tensor, so `--repack` must not change this kernel's workload, and it
  does not - which also means the 26.70 vs 27.29 t/s difference between the two
  rows is machine drift, not a configuration effect. This is a direct measurement
  of the drift floor: about 2% on pp for the same workload half an hour apart.
- The element counts are 3-13% below the static estimate in section 5 (for example
  73 756 measured vs 83 968 predicted per token for the 1B). The static estimate
  counts every tensor whose `vec_dot_type` is `Q8_K` from the GGUF inventory, but
  the runtime decision between the tiled path, the repack path and the plain
  vec_dot path depends on shape and on the graph, and only some of those paths call
  this function. The count measured in situ is authoritative.
- The measured share is 1.03-1.20x the analytical share of section 5 (1.384 vs
  1.263, 6.437 vs 5.778, 4.80-4.91 vs 4.01-4.24, 0.856 vs 0.830, 3.522 vs 3.124),
  i.e. the per-element cost in a real run is 1.03-1.20x the 2.11 ns measured on an
  L1-hot buffer in `simd-bench`. The `rdtsc` overhead (about 25 cycles per call
  against roughly 3400 cycles of work for a 2048-element row) accounts for under 1%
  of that.

**Upper bound on any end-to-end pp gain**: the share itself, reached only if the
kernel became free. For the decode phase the analytical shares of section 5 apply
(0.26-1.65%), so tg is expected to stay inside the noise. Phase 5's "too good"
check uses these numbers: a pp gain above twice the share in the table above cannot
be real.
