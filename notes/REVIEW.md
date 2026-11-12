# REVIEW: the whole change, line by line

The production change is **one function in one file**:

```
$ git diff --numstat upstream/master...HEAD -- ggml/src/ggml-cpu/arch/x86/quants.c
130     3       ggml/src/ggml-cpu/arch/x86/quants.c

$ git diff -U0 upstream/master...HEAD -- ggml/src/ggml-cpu/arch/x86/quants.c | grep -c '^@@'
1
@@ -504,3 +504,130 @@ void quantize_row_q8_1(...)
```

So: the 3 lines that were replaced are the old forwarder, and everything else on
this branch is `notes/` and `tools/simd-experiment/` (test tooling, no effect on
any library). The function now lives at lines 504-633 of
`ggml/src/ggml-cpu/arch/x86/quants.c`.

What was there before:

```c
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_q8_K_ref(x, vy, k);
}
```

## The contract

`quantize_row_q8_K` must produce **byte-identical** output to
`quantize_row_q8_K_ref` for every input. The output is one `block_q8_K` per 256
elements: a `float d`, 256 `int8_t qs` and 16 `int16_t bsums` (292 bytes total,
`qs` at byte offset 4, `bsums` at 260). Because every output byte is discrete, this
is a testable equality, not a tolerance question - and because the same bytes feed
every K-quant and IQ-quant `vec_dot`, matching them exactly means the model output
cannot change at all. `notes/baseline.md` explains why the tolerance is zero and
why that is not available to the pre-existing `q8_0` pair.

## Walkthrough

Line numbers are the committed file's.

### 504-506: signature and block count

```c
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
```

Unchanged semantics: `k % 256 != 0` is a caller contract violation (asserted), the
tail elements beyond `nb*256` are ignored and never read, and `nb` is `int64_t`
like the reference's `int64_t nb`.

### 630-633: the `#else` path

```c
#else
    GGML_UNUSED(nb);
    quantize_row_q8_K_ref(x, vy, k);
#endif
```

x86 without AVX2 gets the reference, exactly as before. The new code is inside
`#if defined(__AVX2__)` and there is no runtime dispatch anywhere, so this function
can never execute an AVX2 instruction on a CPU without AVX2.

### 511-537: the block maximum, taken in the integer domain

```c
        const __m256 sign_bit = _mm256_set1_ps(-0.0f);
        __m256i imax0 = _mm256_setzero_si256();   // four independent chains
        ...
        for (int it = 0; it < QK_K/32; ++it) {
            const __m256i a0 = _mm256_castps_si256(_mm256_andnot_ps(sign_bit, _mm256_loadu_ps(xb + 32*it + 0)));
            ...
            imax0 = _mm256_max_epi32(imax0, a0);
```

`andnot` clears the sign bit, so each lane holds `|x|` **as a bit pattern**. For
non-negative finite floats those patterns order exactly like the values, so an
integer maximum is an exact float maximum - and unlike `VMAXPS` it has no NaN
special case to reason about. Four chains (`imax0..3`) keep the dependency chains
independent; they are merged with `max_epi32` and a 3-step horizontal reduction
(pairwise shuffle, cross-group shuffle, 128-bit lane swap), leaving the maximum in
lane 0.

Why this is not `_mm256_max_ps` + a horizontal max: that is what variant A/B did,
and it cannot distinguish "+infinity present" from "maximum is +infinity", which is
the bug the fuzzer found (`notes/impl.md`, variant D). In the integer domain every
NaN pattern is *above* `+infinity`, which makes the next check free.

### 541-548: the infinite/NaN guard

```c
        if (imax >= 0x7f800000u) {
            quantize_row_q8_K_ref(xb, yb, QK_K);
            continue;
        }
```

`0x7f800000` is `+infinity`; every NaN pattern is larger. So this single compare
sends **any block containing an infinity or a NaN** to the reference. That is
necessary, not defensive: the reference's `nearest_int(iscale*x)` adds `1.5*2^23`
and reads the mantissa of the result, so for a non-finite product it produces a
value that depends on the NaN **payload** (payload `0x400000` gives 0, payload
`0x7fffff` gives 4194303, i.e. 127 after `MIN(127, .)`). The only way to reproduce
that exactly is to run the reference's own code for those blocks. Real activations
are finite, so this branch costs nothing in practice.

### 552-566: find the first element that carries the maximum

```c
        const __m256i imax_b = _mm256_set1_epi32((int) imax);
        int best = -1;
        for (int it = 0; it < QK_K/8 && best < 0; ++it) {
            const __m256i ax = _mm256_castps_si256(_mm256_andnot_ps(sign_bit, _mm256_loadu_ps(xb + 8*it)));
            const unsigned int m = (unsigned int) _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(ax, imax_b)));
            if (m != 0) { best = 8*it + __builtin_ctz(m); }
        }
```

The reference keeps the **first** element with the largest absolute value (`ax > amax`
is a strict compare, so later equal values never replace it), and that element's
*sign* decides `iscale = -127/max`. `movemask` + `ctz` returns the lowest set bit,
i.e. the earliest lane in the earliest vector, i.e. the first element in memory
order. The loop exits as soon as a match is found; the compare is on integers, so
this cannot misfire on `-0.0` (sign cleared) or NaN (already excluded).

### 568-572: the unreachable guard

```c
        if (best < 0) {
            quantize_row_q8_K_ref(xb, yb, QK_K);
            continue;
        }
```

`imax` was computed from this very block, so some element matches and this cannot
be reached. It exists so the function has a defined fallback instead of relying on
the reader to prove unreachability.

### 574-582: `iscale` and the denormal/zero guard

```c
        const float xmax   = xb[best];
        const float iscale = -127.f/xmax;
        if (!isfinite(iscale)) {
            quantize_row_q8_K_ref(xb, yb, QK_K);
            continue;
        }
```

Two different cases reach this guard, both of which the reference treats specially:
a **denormal** maximum makes `-127/xmax` overflow to infinity (the reference then
writes zeros because `iscale*x` is `inf*denormal = inf` -> `nearest_int` -> 127 ->
`MIN(127, 127)`... and its `!amax` early-out does not fire), and an **all zero**
block gives `xmax = 0` and `iscale = -inf`, where the reference takes its `!amax`
path: it zero-fills `qs` and leaves `bsums` **untouched**. Delegating both keeps
those quirks out of this kernel.

Note `xmax` is the *signed* value, so `iscale` carries the sign of the maximum -
just like `-127.f/max` in the reference, which is why the argmax tie rule above
matters.

### 584-604: the quantization pass

```c
            const __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + 64*j +  0), iscalev));
            ...
            const __m256i q0 = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(_mm256_packs_epi32(i0, i1),
                                                                              _mm256_packs_epi32(i2, i3)), unpack);
```

`VCVTPS2DQ` rounds to nearest-even from MXCSR, which is exactly what `nearest_int`
does (magic add of `1.5*2^23` then mantissa subtraction), so for equal products the
integer is equal - this is the load-bearing equivalence of the whole kernel.

There is **no `MIN(127, v)` step**, and none is needed: `|x| <= |max|`, and the two
roundings (one in `-127/xmax`, one in the multiply) keep every product below `127.5`,
so `VCVTPS2DQ` returns at most `127`. `VPACKSSDW` then `VPACKSSWB` narrow *and*
saturate, and saturation at `127` coincides with what `MIN(127, .)` would have done. The pack
instructions work per 128-bit lane, which interleaves the four 8-byte groups, so the
single `unpack` permutation (`0,4,1,5,2,6,3,7`) restores memory order before the
32-byte store.

A reader who wants to break this kernel should attack exactly one assumption here:
that `iscale*x` is always in `[-127, 127]`. That is what the `clamp_edge` corpus and
the fuzzer are for.

### 606-626: `bsums`, the running sums of 16 consecutive bytes

```c
            __m256i t0 = _mm256_maddubs_epi16(ones_u8, q0);
            __m256i t1 = _mm256_maddubs_epi16(ones_u8, q1);
            t0 = _mm256_madd_epi16(t0, ones_i16);
            t1 = _mm256_madd_epi16(t1, ones_i16);
            __m256i s = _mm256_hadd_epi32(t0, t1);
            s = _mm256_hadd_epi32(s, s);
            s = _mm256_permutevar8x32_epi32(s, gather4);
            s = _mm256_packs_epi32(s, s);
            _mm_storel_epi64((__m128i *) (yb->bsums + 4*j), _mm256_castsi256_si128(s));
```

`maddubs` multiplies unsigned bytes by signed bytes; the **constant 1 is the
unsigned operand and the data is the signed operand**, which is the required order.
It sums each pair into an `int16`, `madd` with an all-ones `int16` sums each pair of
those into an `int32` (sums of 4 bytes), and the two `hadd` steps produce sums of 16
bytes in int32 lanes 0, 1, 4 and 5. `gather4` collects those four into the low 128
bits, `packs` narrows them to int16, and one 8-byte store writes 4 `bsums` per
64-element iteration, 16 in total.

Bounds: a pair sum is within +/-256, a group sum within +/-2048, a 16-byte sum
within +/-2048 - all far inside int16/int32 range, so nothing can saturate where the
reference would not.

### 628: `d`

```c
        yb->d = 1/iscale;
```

The same scalar expression as the reference, stored into the same field. `d` is a
plain `float` in `block_q8_K` (`ggml-common.h:372`; `sizeof == 292` is asserted),
so there is no fp16 conversion step in this path at all.

## Invariants this code relies on

1. `k % 256 == 0` (asserted; the callers guarantee it) and nothing outside
   `[0, nb*256)` is read or written.
2. Blocks with an infinity, a NaN, a denormal maximum or a zero maximum are handed
   to the reference, because their results are payload- or quirk-dependent.
3. For every other block `|x[j]| <= |max|`, hence `|iscale*x[j]|` stays below
   `127.5` (worst case `127*(1+2^-24)^2`) and the missing clamp is unreachable.
4. `nearest_int` and `VCVTPS2DQ` agree for every finite product.
5. The first occurrence of the maximum in memory order is the one whose sign is
   used.
6. Every load and store is unaligned-safe; `qs` and `bsums` are never assumed to be
   aligned beyond the struct's own 4 bytes.
7. `_mm256_maddubs_epi16` operand order (unsigned constant first, signed data
   second).

If you want to check one of these, the tools are: `simd-diff --corpus <name>`
(bit-exactness against the reference, with a self-test), `simd-bench --check`
(equality on the timed buffers) and `fuzz-q8k` (random inputs through both
implementations). `notes/verification.md` records what each of them covered and
`notes/impl.md` records the one bug they found.

## What is risky or non-obvious

- The four guards are the *only* thing keeping the kernel bit-exact on hostile
  inputs; they are not defensive coding, they are required for correctness. A
  future reader who removes one to "simplify" reintroduces a real bug (the fuzzer
  found exactly that class in 53 executions).
- The integer-domain maximum is correct **only** because the sign bit is cleared
  first; comparing raw float bit patterns without `andnot` would order negative
  values wrongly.
- `imax` is used both as the maximum and as a NaN/infinity detector. That is
  deliberate and it is why `_mm256_max_ps` is gone; it is not an accident of the
  representation being reused.
- The reference is the oracle even where it does something questionable
  (`nearest_int` on a NaN, the undocumented `bsums` behaviour in its `!amax` path,
  an implementation-defined float-to-int8 cast). Matching it byte for byte is the
  point of the change, so those paths are delegated rather than "fixed".
- The kernel does not use FMA, AVX-VNNI, F16C or AVX-512 even when the compiler is
  allowed to (verified with `objdump` in all six build configurations,
  `notes/verification.md` section 3), so the `__AVX2__` guard is honest.
