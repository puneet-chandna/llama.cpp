# Phase 4: implementation notes

Only one production file changes: `ggml/src/ggml-cpu/arch/x86/quants.c`. The
forwarder at the bottom of the `Q8_K` section is replaced by an AVX2 body; the
`#else` branch still calls `quantize_row_q8_K_ref` for x86 builds without AVX2,
and every degenerate block is delegated to the same reference per block.

Build and check commands used for every variant (from the repository root):

```sh
CM=~/.local/bin/cmake
$CM --build build/release -j 10 --target simd-bench simd-diff llama-bench llama-perplexity
build/release/bin/simd-diff --corpus all --blocks 5000
taskset -c 0,2 build/release/bin/simd-bench --impl both --k 256,512,2048,4096,8192,32768 --reps 25
```

`simd-diff --self-test` is run before every correctness claim. It injects one byte
into the second output buffer and requires the same comparison code to report it
(and to report nothing after the byte is restored), which proves that the
differential tester compares what it says it compares.

## Variant A: per lane first-argmax tracking

Design and the lane-level derivation are in `notes/research.md` 3.7. Summary:

- pass 1 keeps `(amaxv, kbestv)` per 32-bit lane with `VANDNPS`, `VCMPPS` with
  `_CMP_GT_OQ`, `VBLENDVPS` and `VPBLENDVB`;
- the 8 lanes are folded with the key `(amax, kbest, lane)`, which is the same
  order as the element index `8*kbest + lane`, so the winner is the first element
  of the block that reaches the maximum;
- `max` is then just `x[best]`, which is by construction the value the reference
  stores, so the sign rule cannot drift;
- two guards fall back to `quantize_row_q8_K_ref` for the whole block: `best < 0`
  (the reference's `!amax` case: all zero or all NaN) and
  `!(isfinite(xmax) && isfinite(iscale))` (infinite and denormal maxima);
- pass 2 converts, clamps, packs and stores 64 elements per iteration, and
  derives `bsums` from the packed bytes with `VPMADDUBSW` + `VPMADDWD` + two
  `VPHADDD` steps.

### Bug found by the differential tester (and fixed)

The first version of pass 2 used `_mm256_permute4x64_epi64(..., 0xD8)` to undo the
`VPACKSSWB` lane interleave, on the assumption that a qword level permute with
order (0,2,1,3) equals the dword level index vector `(0,4,1,5,2,6,3,7)` that
`quantize_row_q8_0` uses. It does not: the pack instructions interleave at 8-byte
granularity, so a qword swap leaves elements 4..7 holding the values of elements
8..11.

`simd-diff --corpus all --blocks 20000` caught it immediately:

```
corpus=uniform         calls=5000      blocks=5000        mismatches=5000
corpus=wide            calls=5000      blocks=5000        mismatches=4965
corpus=denormal        calls=5000      blocks=5000        mismatches=0
corpus=tie             calls=5000      blocks=5000        mismatches=5000
...
MISMATCH corpus=tie call=0 k=256 blocks=1 x_off=0 y_off=0 byte=8
  region=block_0_qs[4] ref=-125 candidate=-127
```

The inputs of that block are `+36.0` at index 0 (`42100000`) and `+35.5` at index 4
(`420e0000`), with `iscale = -127/36`. `q[4] = -127` is what `iscale * 36` rounds
to, i.e. the element at index 4 had received the value of a different element -
exactly the signature of a one-group shift. Switching to the dword index vector
gave 0 mismatches on all corpora. This is also the reason the corpora that cannot
distinguish a permutation (all-zero, all-NaN, all-Inf, and every block that takes
a guard) showed 0 mismatches before the fix.

### Compiler output (GCC 15.2.0, `-O2 -march=native`, after the fix)

`objdump -d --no-show-raw-insn -M intel` on the object file, function body only
(189 lines total, two loops):

```
    380:	vandnps ymm3,ymm5,YMMWORD PTR [rax]      # |x| = andnot(sign, loadu(x))
    38b:	vcmpgt_oqps ymm0,ymm3,ymm1               # ordered >, false for NaN
    390:	vblendvps ymm1,ymm1,ymm3,ymm0            # amaxv = blend(amaxv, ax, gt)
    396:	vpblendvb ymm0,ymm4,ymm2,ymm0            # kbestv = blend(kbestv, itv, gt)
    39c:	vpaddd ymm2,ymm7,ymm2                    # itv += 1
    3a4:	jne    380                               # pass 1 loop, 32 iterations
    ...
    508:	vmulps ymm9,ymm2,YMMWORD PTR [rdi+0xe0]  # iscale*x
    53d:	vcvtps2dq ymm0,ymm0                       # round to nearest even
    541:	vpminsd ymm9,ymm15,ymm1                   # MIN(127, v)
    58c:	vpacksswb ymm0,ymm0,ymm13                 # int32 -> int16 -> int8
    5a0:	vpermd ymm0,ymm7,ymm0                     # undo the 128 bit lane interleave
    5b8:	vpmaddubsw ymm0,ymm4,ymm0                 # byte pairs, constant 1 unsigned
    5c2:	vpmaddwd ymm0,ymm0,ymm3                   # sums of 4 bytes
    5f8:	vpermd ymm0,ymm11,ymm0                    # gather the 16 byte sums
    601:	vmovq  QWORD PTR [rax-0x8],xmm0          # 4 x int16 bsums
    60a:	jne    508                               # pass 2 loop, 4 iterations
    610:	vmovss xmm0,DWORD PTR [rip+0x0]
    618:	vdivss xmm0,xmm0,xmm7                    # d = 1/iscale
    61c:	vmovss DWORD PTR [r14],xmm0              # y[i].d
```

Checks performed on the census of the function body:

| check | result |
|---|---|
| FMA / AVX-VNNI / F16C / AVX-512 opcodes (`vfmadd*`, `vpdp*`, `vcvtph*`, `kmov*`, `vpternlog*`) | none |
| `vcvtps2dq`, `vpmaddubsw`, `vpmaddwd` present | yes (8, 2, 2 in the pass 2 loop body) |
| per-element scalar `cvtsi2ss`/`movd` sequences | none |
| `call` instructions inside the body | 2, both the degenerate-block calls to `quantize_row_q8_K_ref` (`45b`, `626`) |
| `%ymm` spills to the stack inside either hot loop | none (the only `[rsp+..]` accesses are the `axlane`/`kblane` arrays between the two loops) |

The 2 calls are outside the hot loops and per block at most; they are taken only
when a block is all-zero, all-NaN, or has an infinite or denormal maximum.

### Instruction mix and expected throughput

| loop | uops per iteration | iterations | elements per iteration |
|---|---|---|---|
| pass 1 | 8 (load, andnot, cmp, 2 blends, add, loop bookkeeping) | 32 | 8 |
| pass 2 | ~47 (8 mul, 8 cvt, 8 min, 7 pack, 4 permd, 2 store, 4 madd, 2 hadd, 1 storeq) | 4 | 64 |

Roughly 440 uops per 256-element block, i.e. about 1.7 uops per element. Pass 1 is
the larger half, which is what variant B attacks.

## Variant B: reduction-first maximum plus an exact repair pass

Replaces pass 1 and the lane fold with:

- one `VMAXPS` chain over `|x|` (`VANDNPS` + `VMAXPS`), then two shuffles and
  `_mm_max_ss` for the horizontal maximum - 5 uops per 8 elements instead of 8;
- a **repair pass** that scans the 32 vectors again, computes
  `movemask(cmp_eq(|x|, amax))` and takes `tzcnt` of the first non-zero mask, so
  `best` is the first element of the block with `|x| == amax` - which is exactly
  the reference's first argmax, hence the same `max` and the same `iscale` sign.

Two exactness details that the sketch in the plan does not mention, both needed:

1. **Operand order of `VMAXPS`.** The new value is the first operand and the
   running maximum the second: `_mm256_max_ps(ax, amaxv)`. `VMAXPS` returns its
   *second* operand when either input is NaN (verified on this machine:
   `max(3.0, NaN) = NaN` but `max(NaN, 3.0) = 3.0`). With this order a NaN can
   never replace the running maximum, so the chain reproduces the reference's
   "NaN never satisfies `ax > amax`" rule with **zero** extra instructions. In the
   other order, a NaN would *reset* the accumulator and the block maximum would
   come out too small for any block that mixes NaN with finite values; the
   `mixed_nonfinite` corpus is what would catch it.
2. The maximum is therefore always finite in every lane, so the horizontal
   reduction needs no NaN handling either, and an all-NaN block ends with
   `amax = 0` and no match in the repair pass, which takes the reference path.

Codegen (161 lines for the whole function, GCC 15.2.0, `-O2 -march=native`):

```
     360:	vandnps ymm2,ymm1,YMMWORD PTR [rax]   # |x|
     36b:	vmaxps ymm0,ymm2,ymm0                  # new value first, NaN neutral
     36f:	jne    360                             # pass 1 loop, 32 iterations
     ...
     395:	vandnps ymm2,ymm1,YMMWORD PTR [rsi]    # repair pass
     399:	vcmpeqps ymm2,ymm2,ymm0
     39e:	vmovmskps edx,ymm2
     3a4:	je     438                            # no match in this vector
     3aa:	tzcnt  edx,edx                         # first matching lane
     ...
     591:	jne    499                             # pass 2 loop, unchanged
```

| check | result |
|---|---|
| FMA / AVX-VNNI / F16C / AVX-512 opcodes | none |
| `call` instructions in the body | 1 (the single degenerate-block call to `quantize_row_q8_K_ref`, reached only for all-zero, all-NaN, infinite or denormal maxima) |
| stack references anywhere in the body | 2, both a spilled pointer outside all three loops; no `%ymm` spill in any loop |
| `tzcnt` | present (BMI1, available on this CPU); without `-mbmi` GCC emits `bsf`, which is equivalent here because the mask is non-zero |

## Measurements

`simd-bench` times the two implementations **alternately inside each repetition**
and reports the median of the per-repetition ratios. That matters on this host:
the same reference binary measured 2.11 ns per element in one session and
1.61 ns per element in another (a factor of 1.31, `scaling_governor = powersave`,
`-t 2`, pinned to cores 0 and 2), so absolute times are only comparable inside one
session. The paired ratio is stable across sessions, the absolute numbers are not.

GCC 15.2.0, `-O2 -march=native`, `--reps 41`, `--impl both`, one call per sample:

| k | ref ns | candidate ns | paired ratio | ratio sd |
|---|---|---|---|---|
| 256 | 511 | 137 | 3.72x | 0.09 |
| 512 | 1011 | 248 | 4.06x | 0.04 |
| 1024 | 1712 | 399 | 4.28x | 0.47 |
| 2048 | 3389 | 781 | 4.34x | 0.04 |
| 3072 | 4961 | 1133 | 4.38x | 0.43 |
| 4096 | 6598 | 1510 | 4.37x | 0.02 |
| 8192 | 13193 | 3001 | 4.40x | 0.13 |
| 32768 | 52599 | 11916 | 4.41x | 0.06 |

The two `k` values that occur in the models used for the end-to-end runs are
`k = 2048` (1B and 7B `n_embd`) and `k = 3072` (3B `n_embd`): **4.34x and 4.38x**.

The candidate reaches 0.364 ns per element at large `k`, i.e. about 1.6 cycles per
element at 4.4 GHz against roughly 1.7 uops per element of static instruction
count, so it is not simply ALU-throughput bound. Pass 1 (8 uops per 8 elements, and
a serial `amaxv` -> `VCMPPS` -> `VBLENDVPS` dependency of about 6 cycles per
iteration, 32 iterations per block) is the prime suspect, and it is what variant B
replaces.

Correctness at this stage: `simd-diff --corpus all --blocks 5000` = 0 mismatches on
all 15 corpora, `simd-diff --self-test` passes, `simd-bench --check` passes at every
`k`. Full verification is phase 5.

## Variant A vs variant B

Two builds in one session, each measuring its own paired ratio against the
reference (variant A from a `git worktree` at the phase 4a commit, variant B from
the working tree), three rounds, 41 repetitions each:

| k | variant A ratio vs ref | variant B ratio vs ref | B/A |
|---|---|---|---|
| 2048 | 4.546x, 4.508x, 4.545x | 6.291x, 6.648x, 6.343x | 1.38x, 1.47x, 1.40x |
| 3072 | 4.539x, 4.473x, 4.546x | 6.038x, 6.325x, 5.973x | 1.33x, 1.41x, 1.31x |

The reference times inside each round overlap between the two builds (4550-5772 ns
versus 4720-5524 ns at k=2048), so the two builds saw comparable machine states,
and the ratio is in any case measured paired within each process.

Variant B wins by about 1.35-1.45x. That is consistent with variant A being
latency-bound rather than throughput-bound: A carries a `VCMPPS` + `VBLENDVPS`
chain of about 6 cycles through all 32 iterations of pass 1, while B's `VMAXPS`
chain is 4 cycles per iteration and is the only chain, and its repair pass exits
after the first matching vector (half a block on average).
