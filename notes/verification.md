# Phase 5: verification

Kernel under test: the committed kernel (`quantize_row_q8_K`, variant D),
`ggml/src/ggml-cpu/arch/x86/quants.c`, commit `f12457c14`. Every number in this
file was produced after the integer-domain fix described in `notes/impl.md`
("Variant D"). Sections marked "pre-fix" are kept for history.

Baseline ("before") side: the pristine worktree at `5fac66808`, a clean checkout of
this branch's base commit with no changes to `ggml/src`. Its
`sha256(libggml-cpu.so) = 17e8c6a9d07633d30182c3a1f76339e7` and its
`quantize_row_q8_K` is `jmp quantize_row_q8_K_ref@plt`, so the "before" side
executes the scalar reference by construction (verified with `objdump`, section 8).

## 1. Differential testing (bit-exactness)

`simd-diff` compares the whole 292-byte `block_q8_K` plus 32 sentinel bytes on each
side of the output region, so it also fails on an out-of-bounds write. Zero
tolerance: any differing byte is a failure (`notes/baseline.md` section 1 explains
why zero is the right tolerance here, and why the same zero tolerance is not
available to the pre-existing `q8_0` pair).

All runs below reported **0 mismatches**.

| run | build | corpora | blocks per corpus | total blocks |
|---|---|---|---|---|
| release sweep, per corpus | `build/release` | 16 | 10 000 000 (28 000 000 for `odd_k`) | 178 000 000 |
| compiler/flag matrix | `build/gcc-o2`, `build/clang-o2`, `build/clang-o3`, `build/minfeat` | 16 each | 1 000 000 (2 800 000 for `odd_k`) | 71 200 000 |
| sanitizer build | `build/asan` (ASan + UBSan) | 16 | 200 000 (560 000 for `odd_k`), `--misalign 4` | 3 560 000 |
| seed for the fuzzer | `build/release` | 16 | 4 blocks (aligned and tail variants) | 128 |

Totals on the committed kernel: **252 760 000 blocks, i.e. 6.47e10 quantized
elements**, all bit-identical to the reference, recorded in 96 corpus-runs of the
logs above plus the 128-block seed generation. For comparison, the pre-fix kernel
was measured over 238 560 020 blocks (6.11e10 elements) with the same result; the
extra 14 199 980 blocks here are the new `nan_mix` corpus.

The plan asked for 1e8 blocks (2.56e10 elements). `--blocks` is per corpus, so the
release sweep alone is 1.78e8 blocks (4.6e10 elements), about 1.8x the requested
volume, and the compiler matrix adds 7.1e7 more.

Corpus coverage (16): `uniform`, `wide` (all normal exponents), `denormal` (all
subnormal, so the block maximum is subnormal and `iscale` overflows), `tie` (block
maximum repeated many times with both signs - the argmax tie rule), `spike`, `zero`
(signed zeros), `allzero`, `extreme` (`+-FLT_MAX`, `+-FLT_MIN`, `+-2^-149`), `inf`,
`nan` (quiet, signalling, negative, varying payloads), `mixed_nonfinite`,
`clamp_edge` (`iscale == 1.0f` exactly, elements at `+-127.0`, `+-126.9999` and
every exact half-integer), `unaligned` (x and y at 1..31 byte offsets), `odd_k`
(`k` in {0, 256, 640, 769, 2303}, i.e. `k % 256 != 0` for three of them),
`random_mix`, and `nan_mix` (finite values with a **finite** maximum plus NaNs
scattered in - the shape that the fuzzer found, section 2).

Harness self-checks, run before every claim above:

```
$ build/release/bin/simd-diff --self-test
self-test: PASS (clean=clean injected=detected reverted=clean)
$ build/release/bin/simd-diff --impl ref --corpus all --blocks 200 --quiet
total mismatches: 0                      # the reference against itself
$ build/release/bin/simd-bench --impl both --k 2048,4096 --reps 5 --check
check: OK k=2048 rows=1 bytes=2336
check: OK k=4096 rows=1 bytes=4672
```

`--self-test` is the answer to "is the differential test actually comparing?" - it
injects one byte into the second output buffer and requires the same comparison
code to report it, then restores the byte and requires a clean comparison again.

## 2. Fuzzing

Harness: `tools/simd-experiment/fuzz-q8k.cpp`, Clang 21.1.8,
`-fsanitize=fuzzer,address`, built against the instrumented `ggml` libraries (the
harness links `libggml-cpu.so` built with `-fsanitize=fuzzer-no-link,address`, so
the sanitizer and coverage instrumention covers the kernel itself, not only the
harness). The input bytes select `k` (including non-multiples of 256), a source and
destination byte offset, and the float payload; the harness quantizes with both
implementations and calls `abort()` on any byte difference. Driven by
`notes/run_fuzz.sh`.

**The first campaign found a real bug after 53 executions.**

```
MISMATCH k=1024 nb=4 x_off=3 y_off=12 first_byte=1125 ref=0x7f candidate=0x80
==369611== ERROR: libFuzzer: deadly signal
artifact_prefix='./'; Test unit written to ./crash-d109a2f0f1b19126d76d413da5cfb62ef46da9ea
```

A block with six NaNs and a finite maximum of 127 was stored as `-128` where the
reference stores `127`. Variants A, B and C were all affected. Root cause, the fix
and the regression evidence are in `notes/impl.md` ("Variant D"). The corrupting
input (5914 bytes) is archived as `notes/fuzz-seeds/fuzz-q8k-nan-finite-mix.bin`
(`sha256 b12ce071f81afed24060f24f...`) and `notes/run_fuzz.sh` copies every archived
seed into the corpus, so it is exercised by every campaign.

Campaigns (both with `-max_total_time=900 -max_len=8192 -rss_limit_mb=2048`,
`-max_len` forces the mutator across the `k`/offset/payload boundaries):

| campaign | kernel | result |
|---|---|---|
| 1 | variant C (pre-fix) | aborted after 53 executions on the `nan_mix` shape |
| 2 | committed kernel | **16 007 485 executions in 901 s (17 766 exec/s), no failures** |

Campaign 2 details: `cov: 43` covered edges, `ft: 148` features, corpus 44 units
(68 Kb) from 37 seeds, 507 new units added, peak RSS 425 MB. The archived crash
input replays clean on the committed kernel:

```
$ build/fuzz/fuzz-q8k -runs=1 notes/fuzz-seeds/fuzz-q8k-nan-finite-mix.bin ; echo $?
0
```

## 3. Assembly fingerprint per configuration

`objdump -d --no-show-raw-insn -M intel` on the `quantize_row_q8_K` body of each
built library. On the committed kernel `vpmaxsd` must be present (the integer
maximum), `vmaxps` and `vpminsd` must be 0 (no float maximum, no clamp), there must
be exactly one `tzcnt`/`bsf` (the repair pass), and no FMA / AVX-VNNI / F16C /
AVX-512 opcode anywhere:

| build | vpmaxsd | vmaxps | vpminsd | tzcnt/bfs | calls |
|---|---|---|---|---|---|
| release (gcc, `-O2 -march=native`) | 38 | 0 | 0 | 1 | 1 |
| gcc-o2 | 10 | 0 | 0 | 1 | 1 |
| clang-o2 | 3 | 0 | 0 | 1 | 1 |
| clang-o3 | 3 | 0 | 0 | 1 | 1 |
| minfeat (AVX2 only, no `-march=native`) | 38 | 0 | 0 | 1 | 1 |
| asan (RelWithDebInfo) | 10 | 0 | 0 | 1 | 63 (sanitizer instrumentation) |

The single `call` is the degenerate-block call to `quantize_row_q8_K_ref`. No `[rsp]`
access appears inside any of the three loops in any configuration (checked for
gcc-o2 and clang-o3 as well). `vpmaddubsw`/`vpmaddwd` (the `bsums` reduction) and
`vpcmpeqd` + `vpmovmskb` (the repair pass) are present in all of them.

The independent review asked for one comment to be reworded (`notes/audit.md`,
finding F1). After that comment-only edit, all six configurations were rebuilt and
re-measured: every fingerprint above is unchanged and each configuration reports 0
mismatches on a 3 200 000-block sweep, and the fuzz library replays the archived
crash input cleanly. So the verified machine code is the committed machine code
(`build/logs/phase6_rebuild_check.txt`).

## 4. Minimum instruction set

The plan asked for a `-mavx2`-only build as proof that the kernel needs nothing
beyond AVX2. `build/minfeat` is that build for the kernel: it enables AVX2, disables
AVX-512 and AVX-VNNI, and does not use `-march=native`, and it compiles this file
and runs the full 16-corpus matrix with 0 mismatches (section 1). The file as a
whole cannot be built with FMA disabled, for pre-existing reasons outside this
change: `arch/x86/quants.c` uses `_mm256_fmadd_ps` unguarded in the vector dot
product helpers (lines 728 and 730 in the committed file), so `-DGGML_FMA=OFF` fails
to compile the file, and `-DGGML_F16C=OFF` additionally fails in
`arch/x86/repack.cpp` (a `const ggml_half *` passed where the non-F16C helper wants
`ggml_fp16_t *`). Both are reported as observations in the final report.

An earlier draft of this section claimed the file "compiles with
`-mavx2 -mno-fma`" on the basis of a probe whose output was filtered by a grep
pattern that did not match the actual diagnostics (they are attributed to
`fmaintrin.h`, the header that defines the FMA intrinsics). That claim was wrong and
is corrected here. What the probe does show, once read in full: with `-mno-fma` and
`-mavx2`, GCC reports errors **only** at those two pre-existing `_mm256_fmadd_ps`
sites and in no other function - and GCC reports every error it encounters in a
translation unit rather than stopping at the first, so the absence of an error
inside `quantize_row_q8_K` (lines 504-630) means the kernel itself needs no more
than AVX2. That is corroborated by two independent facts:

1. The built kernel contains **no FMA / VNNI / F16C / AVX-512 opcode even when the
   compiler is allowed to use them** (section 3, `-march=native` enables all of
   them), which is the property that matters: the code does not silently depend on
   a wider ISA than the `#if defined(__AVX2__)` guard claims.
2. `build/minfeat` (AVX2 enabled, AVX-512 and AVX-VNNI disabled, no
   `-march=native`) runs the full matrix with 0 mismatches.

## 5. Alignment: what UBSan reports, and why the real callers are unaffected

With the tester's deliberate 1-byte misalignment (`--misalign 1`), the ASan+UBSan
build reports `-fsanitize=alignment` diagnostics from **both** implementations:

```
ggml/src/ggml-quants.c:2793,2795,2808            load of misaligned address ... type 'const float'   (reference)
ggml/src/ggml-quants.c:2809,2814,2816,2818       member access within misaligned address ... 'block_q8_K'  (reference)
ggml/src/ggml-cpu/arch/x86/quants.c:573          load of misaligned address ... type 'const float'   (committed kernel: xb[best])
ggml/src/ggml-cpu/arch/x86/quants.c:611,612,625  member access within misaligned address ... 'block_q8_K'  (committed kernel: yb->qs, yb->bsums)
ggml/src/ggml-cpu/arch/x86/quants.c:628          member access within misaligned address ... 'block_q8_K'  (committed kernel: yb->d)
```

Reading of that result:

- These are pedantic C-level diagnostics, not hardware faults. Every vector access
  in the kernel is `_mm256_loadu_ps` / `_mm256_storeu_si256` / `_mm_storel_epi64`;
  the flagged lines are the two scalar accesses (`xb[best]`, `yb->d`) and the
  base-of-member expressions. The reference has more of the same diagnostics, and
  the pre-existing AVX2 `quantize_row_q8_0`/`q8_1` kernels use the same idiom
  (`_mm256_storeu_si256((__m256i *)y[i].qs, i0)`).
- **No real caller can hit this.** The alignment contract in the CPU backend is
  4 bytes, not 32: `params->wdata` is aligned and rows advance by
  `ne10_block_start * sizeof(block_q8_K)` = a multiple of 292, which is a multiple
  of 4, so both `x` and `y` are always 4-byte aligned in `ggml-cpu.c`,
  `tiled/tiled.cpp` and the flash-attention path. A 4-byte-aligned `block_q8_K *`
  satisfies the struct's own alignment requirement, so none of the above is
  undefined for it.
- The property that *does* matter for `y` - that `qs` and `bsums` are never assumed
  to be 32- or 16-byte aligned, since `qs` sits at byte offset 4 inside a 292-byte
  struct - is exercised by **every** run in section 1, because even with perfectly
  aligned buffers the 32-byte stores always land at offset 4 mod 32 and the 8-byte
  `bsums` stores at offsets that are multiples of 8 plus 4.
- Consequently the sanitizer matrix run in section 1 uses `--misalign 4`, which is
  the realistic case (4-byte aligned, not 32-byte aligned), and it produces **no**
  alignment diagnostics at all while still exercising unaligned 32-byte stores.

## 6. Test suites

| suite | build | result |
|---|---|---|
| `test-quantize-fns` | `build/asan` (ASan + UBSan) | exit 0 |
| `test-backend-ops test -b CPU -o MUL_MAT` | `build/release` | 2794/2794 passed, `Backend CPU: OK` |
| `ctest -j2` (full suite) | `build/asan` | 61/67 passed (91%), 6 failures |
| `test-quantize-perf` | `build/release` | not applicable: the tool has no `q8_K` case (`grep -c q8_K tests/test-quantize-perf.cpp` = 0), so it cannot measure this kernel |

The 6 `ctest` failures are **identical** to the failures of the same suite built from
the pristine `5fac66808` worktree with the same flags (same 6 test numbers, same
messages, same 61/67 count):

```
$ diff <(grep '(Failed)' build/logs/phase5_ctest.log    | sed 's/^ *//' | sort) \
       <(grep '(Failed)' build/logs/pristine_ctest.log  | sed 's/^ *//' | sort)
IDENTICAL
16 - test-tokenizers-ggml-vocabs   37 - test-jinja        38 - test-jinja-py
39 - test-conversion-fp8           40 - test-kv-cache-scales  52 - test-llama-mmap
```

That is the attribution the plan asked for: these are pre-existing or
environmental failures (missing Python modules, missing tokenizer assets, a
`test-llama-mmap` behaviour difference), not caused by this change.

## 7. Static analysis

| tool | version | scope | result |
|---|---|---|---|
| `clang-tidy` | 21.1.8 | `-checks=-*,bugprone-*,clang-analyzer-*,cert-*,performance-*` on `arch/x86/quants.c`, `-header-filter=.*` | **0 findings in the changed file**; 14 findings in `ggml/src/ggml-cpu/simd-mappings.h` (`bugprone-macro-parentheses` in pre-existing macros), 68 818 suppressed in non-user code |
| `scan-build` | 21.1.8 | `clang-o3` build of the `ggml-cpu` target with the file force-rebuilt | `No bugs found` (exit 0) |
| `cppcheck` | 2.22.0 (built from source) | `--project=build/clang-o3/compile_commands.json --file-filter='*/arch/x86/quants.c' --enable=warning,style,performance,portability` | **0 findings in the changed file**; 8 findings in `ggml/src/ggml-impl.h` (pre-existing header) |

`cppcheck` was not installed and could not be installed (no `sudo`); it was built
from source in `~/opt/src/cppcheck-2.22.0` with the pip-installed CMake, and the
build succeeded, so the tool is covered rather than skipped.

## 8. "Too good" check

The end-to-end gain is checked against the analytical share before it is reported:

- **The "before" side really executes the reference.** `objdump` on the pristine
  `libggml-cpu.so` shows `quantize_row_q8_K` = `jmp quantize_row_q8_K_ref@plt`, and
  the differential runs use that same library.
- **The differential test really compares.** `--self-test` injects a single byte into
  the second buffer and requires the comparison to report it (section 1).
- **The kernel is called on the paths the benchmarks exercise.** The
  `--no-repack` ("`-nr`") end-to-end runs exist precisely to bracket the path mix:
  with default flags the Q4_K tensors of a Q4_K_M model are served by the repack
  kernels (which do not use this function), and with `-nr` every matmul goes
  through it. Both are measured and reported side by side.
- **The microbenchmark cannot be a build-level artifact.** `simd-bench --impl both`
  measures the reference and the candidate **in the same binary**, alternating per
  repetition, so a global build effect cancels in the paired ratio.
- **The two end-to-end binaries differ only in this function, by construction and by
  build.** Outside `notes/` and `tools/` the branch changes exactly one file
  (`ggml/src/ggml-cpu/arch/x86/quants.c`, +130/-3) and adds one line to
  `tools/CMakeLists.txt` that builds the new tool executables; `nm -D` on
  `libggml-cpu.so`, `libggml-base.so` and `libllama.so` finds none of their symbols,
  and the two
  release configurations are flag-identical: every relevant `CMakeCache` entry
  (`CMAKE_BUILD_TYPE`, `CMAKE_{C,CXX}_FLAGS_RELEASE = -O3 -DNDEBUG`, `GGML_NATIVE`,
  `GGML_CPU_REPACK`, `GGML_LTO`, `GGML_OPENMP`, `GGML_BACKEND_DL`,
  `GGML_CPU_ALL_VARIANTS`) matches, and the `compile_commands.json` entry for
  `arch/x86/quants.c` is identical apart from the source path. So a whole-binary
  codegen difference cannot explain an end-to-end delta.
- **Single-repetition end-to-end numbers are not trustworthy on this host.** A
  1-repetition smoke run of the interleaved harness (used only to test the harness,
  not recorded as a result) gave `after/before` = 1.068 on 1B pp512, roughly five
  times the measured 1.384% share for that configuration - which is why the reported
  end-to-end numbers come from 5 paired repetitions and are read as paired ratios,
  and why any result above twice the measured share is investigated rather than
  reported.
- The measured pp/tg deltas are then compared with `share x kernel speedup` from
  `notes/baseline.md`, and the report says explicitly when a delta is below this
  machine's noise floor instead of presenting it as a win.

## 9. End-to-end

### 9.1 Perplexity (deterministic, so timing-independent)

Protocol: `llama-perplexity -c 512 -t 2` pinned to cores 0 and 2, same file, same
chunk count, "before" binary = the pristine `build/release` (sha256 `17e8c6a9...`),
"after" binary = `build/release` (sha256 `a8399ac2...`, which is the kernel this
section verifies). The value does not depend on timing - it is a deterministic
reduction - so this part needs no quiet machine, and that is why it was run while
other work was happening.

| model | chunks | before reps | before value | after reps | after value |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct Q4_K_M | 16 | 5 | 16.4452 +/- 0.74551 | 2 | 16.4452 +/- 0.74551 |
| Llama-3.2-3B-Instruct IQ4_XS | 16 | 5 | 12.4257 +/- 0.54566 | 2 | 12.4257 +/- 0.54566 |
| Qwen2.5-7B-Instruct Q4_K_M | 8 | 3 | 6.7962 +/- 0.39313 | 1 | 6.7962 +/- 0.39313 |

All ten before-repetitions and all five after-repetitions printed the same value to
every printed digit, so the delta is exactly zero on all three models. That is the
expected consequence of section 1 (bit-exact kernel -> bit-identical activations ->
bit-identical forward pass) and it satisfies the plan's acceptance criterion in its
strictest form. The `+/-` is llama-perplexity's standard error across chunks, not
run-to-run variation (which is zero here).

Raw: `build/logs/ppl_before-pristine_<model>_<rep>.log`,
`build/logs/ppl_after-fixed_<model>_<rep>.log`,
`build/logs/ppl_after_summary.txt`. Repetitions for the after side were reduced to
2/2/1 because the before side had already established determinism with 5/5/3
identical repetitions.

### 9.2 Throughput (llama-bench), paired interleaved

Protocol: `notes/run_ab_interleaved.sh` runs the two binaries alternately, one
`llama-bench` repetition at a time, both pinned to cores 0 and 2 with `-t 2`
(`-r 1` per invocation, 5 paired repetitions per cell, both repack settings). Raw
data: `build/logs/ab_interleaved.csv` and `build/logs/ab_interleaved_rpk0.csv`,
run logs `build/logs/quiet_window.txt` and `build/logs/quiet_window_rpk0.txt`. The
tables are in `notes/REPORT.md` section 6.2; this section records what the numbers
mean for the method.

**Why cross-session numbers are not comparable.** The 3B IQ4_XS model has no
repackable tensor (nothing changes with `--repack 0`), yet its warm after-side
median moved from 26.85 t/s (baseline session) to 30.94 t/s (this session) with
identical element and cycle counts; the same pristine "before" binary measured
71.28 t/s (baseline session) vs 89.56 t/s (this session) on 1B pp512. Session-level
machine state moves this host by 15-25%, which is 3-20x the effect being measured.
Only the paired ratios are usable.

**Why the paired ratios still cannot resolve the effect.** The 3B's `--repack 0` and
`--repack 1` runs are literally the same workload through the same kernel (IQ4_XS),
so comparing those two cells is a control with no expected difference. Measured:
tg128 -2.10% (se 1.57%) vs +0.99% (se 3.41%); pp512 +0.44% (se 1.14%) vs +1.91%
(se 1.05%). The sign of the difference flips between two runs of the same work, so
the method's real resolution is about +/-3%, not the ~1-2% the per-cell standard
errors suggest, and no cell's delta (range -2.1% to +2.7%) is resolved.

**What the paired data does show.** Nothing that survives the control above, so the
throughput claim is not made. Recorded for the record: the four largest predicted
cells (+5.81%, +4.45%, +4.35%, +3.18% from cycle share x kernel speedup) measured
-0.44%, +1.91%, +0.44%, +1.86%, i.e. the cycle-share arithmetic over-predicts the
wall-clock gain by 1-6 standard errors. The cause is not identified; frequency and
power state are not observable here (`perf` blocked, no RAPL access), and a 15 W
part can trade saved cycles for sustained-clock headroom.

**What is not affected by any of this.** The kernel-level speedup (10.29x at
`k=2048`, 10.59x at `k=3072`, paired, same binary), the bit-exactness (section 1),
and the perplexity equality (section 9.1) are all independent of the throughput
question.

Tooling fixes made while producing this data (both in
`notes/run_ab_interleaved.sh`, committed with this section): the markdown parser now
reads the last two non-empty fields instead of fixed column positions, because
`llama-bench` adds an `rpk` column when `--repack` is not the default (the first
attempt silently produced an empty CSV for `rpk=0`), and `RP_FILTER` selects a
single repack setting. Two failed attempts are visible in
`build/logs/quiet_window_rpk0.txt` history: the first was killed for an unrelated
tool deadline, the second produced no rows before the parser fix.
