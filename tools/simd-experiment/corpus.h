// Deterministic adversarial input corpora for the q8_K differential tester and
// for the libFuzzer seed corpus. Header-only so simd-diff and fuzz-q8k share
// exactly one definition of every corpus.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#ifndef GGML_COMMON_DECL
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#endif

enum class Corpus {
    Uniform,
    Wide,
    Denormal,
    Tie,
    Spike,
    Zero,
    AllZero,
    Extreme,
    Inf,
    Nan,
    MixedNonfinite,
    NanMix,
    ClampEdge,
    RandomMix,
    Unaligned, // same values as Uniform, caller applies 1..31 byte pointer offsets
    OddK,      // RandomMix values, caller uses k values that are not multiples of 256
    Count,
};

struct corpus_entry {
    const char * name;
    Corpus       id;
};

static const corpus_entry CORPUS_TABLE[] = {
    { "uniform",         Corpus::Uniform         },
    { "wide",            Corpus::Wide            },
    { "denormal",        Corpus::Denormal        },
    { "tie",             Corpus::Tie             },
    { "spike",           Corpus::Spike           },
    { "zero",            Corpus::Zero            },
    { "allzero",         Corpus::AllZero         },
    { "extreme",         Corpus::Extreme         },
    { "inf",             Corpus::Inf             },
    { "nan",             Corpus::Nan             },
    { "mixed_nonfinite", Corpus::MixedNonfinite  },
    { "nan_mix",         Corpus::NanMix          },
    { "clamp_edge",      Corpus::ClampEdge       },
    { "unaligned",       Corpus::Unaligned       },
    { "odd_k",           Corpus::OddK            },
    { "random_mix",      Corpus::RandomMix       },
};

static inline int corpus_count(void) { return (int) (sizeof(CORPUS_TABLE) / sizeof(CORPUS_TABLE[0])); }

// xorshift64*, deterministic and portable
struct Rng {
    uint64_t s;

    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ull) {}

    uint64_t next(void) {
        uint64_t x = s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        s = x;
        return x * 0x2545f4914f6cdd1dull;
    }

    uint32_t next_u32(void) { return (uint32_t) (next() >> 32); }

    // [0,1)
    float next_unit(void) { return (float) (next() >> 40) * (1.0f / 16777216.0f); }

    // [-1,1)
    float next_sym(void) { return 2.0f * next_unit() - 1.0f; }

    uint32_t below(uint32_t n) { return n ? next_u32() % n : 0; }
};

static inline float f_from_bits(uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

static inline uint32_t f_to_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

static inline Corpus corpus_gen_for(Corpus c) {
    // corpora that only change how the driver calls the functions
    switch (c) {
        case Corpus::Unaligned: return Corpus::Uniform;
        case Corpus::OddK:      return Corpus::RandomMix;
        default:                return c;
    }
}

// Fill exactly n <= QK_K values of one block. Every generator is written so the
// same rule applies for any n, which keeps k != QK_K multiples meaningful.
static void corpus_fill_block(Corpus c, float * x, int64_t n, Rng & rng) {
    switch (corpus_gen_for(c)) {
        case Corpus::Uniform:
            for (int64_t j = 0; j < n; ++j) {
                x[j] = rng.next_sym();
            }
            break;
        case Corpus::Wide:
            // full normal exponent range, random mantissa and sign
            for (int64_t j = 0; j < n; ++j) {
                const uint32_t exp  = 1 + rng.below(254);
                const uint32_t mant = rng.next_u32() & 0x007fffffu;
                const uint32_t sign = rng.next_u32() & 0x80000000u;
                x[j] = f_from_bits(sign | (exp << 23) | mant);
            }
            break;
        case Corpus::Denormal:
            // every value subnormal, so the block maximum is subnormal too
            for (int64_t j = 0; j < n; ++j) {
                const uint32_t mant = (rng.next_u32() & 0x007fffffu) | 1u;
                const uint32_t sign = rng.next_u32() & 0x80000000u;
                x[j] = f_from_bits(sign | mant);
            }
            break;
        case Corpus::Tie: {
            // the block maximum repeats at many positions with both signs, so any
            // argmax rule other than "first index wins" changes the sign of iscale
            const float m = 1.0f + (float) rng.below(64);
            const bool  flip = rng.below(2) != 0;
            for (int64_t j = 0; j < n; ++j) {
                const int64_t r = j % 8;
                float v;
                if (r == 0 || r == 3 || r == 5) {
                    v = m;
                } else if (r == 1 || r == 6) {
                    v = -m;
                } else if (r == 2) {
                    v = -(m - 0.5f);
                } else {
                    v = m - 0.5f;
                }
                x[j] = flip ? -v : v;
            }
            break;
        }
        case Corpus::Spike: {
            static const float big_tab[5] = { 1.0e-30f, 1.0f, 1.0e10f, 1.0e30f, 3.4028235e38f };
            static const float small_tab[4] = { 1.1754944e-38f, 1.0e-20f, 1.0e-5f, 0.5f };
            const float big = big_tab[rng.below(5)];
            const float sgn = rng.below(2) ? -1.0f : 1.0f;
            const int64_t at = (int64_t) rng.below((uint32_t) n);
            for (int64_t j = 0; j < n; ++j) {
                x[j] = small_tab[rng.below(4)];
                if (rng.below(2)) {
                    x[j] = -x[j];
                }
            }
            x[at] = sgn * big;
            break;
        }
        case Corpus::Zero:
            // signed zeros
            for (int64_t j = 0; j < n; ++j) {
                x[j] = rng.below(2) ? -0.0f : 0.0f;
            }
            break;
        case Corpus::AllZero:
            for (int64_t j = 0; j < n; ++j) {
                x[j] = 0.0f;
            }
            break;
        case Corpus::Extreme: {
            static const float tab[8] = {
                 3.4028235e38f, -3.4028235e38f,
                 1.1754944e-38f, -1.1754944e-38f,
                 1.4012985e-45f, -1.4012985e-45f,
                 1.0e38f, -1.0e38f,
            };
            for (int64_t j = 0; j < n; ++j) {
                x[j] = tab[rng.below(8)];
            }
            break;
        }
        case Corpus::Inf:
            switch (rng.below(4)) {
                case 0:
                case 1: {
                    const float inf = rng.below(2) ? -INFINITY : INFINITY;
                    for (int64_t j = 0; j < n; ++j) {
                        x[j] = rng.next_sym();
                    }
                    x[rng.below((uint32_t) n)] = inf;
                    break;
                }
                case 2:
                    for (int64_t j = 0; j < n; ++j) {
                        x[j] = rng.below(2) ? -INFINITY : INFINITY;
                    }
                    break;
                default: {
                    for (int64_t j = 0; j < n; ++j) {
                        x[j] = rng.next_sym();
                    }
                    x[0] = INFINITY;
                    x[1 % n] = -INFINITY;
                    x[2 % n] = f_from_bits(0x7fc00000u);
                    break;
                }
            }
            break;
        case Corpus::Nan:
            // quiet, signalling and negative NaNs with varying payloads
            for (int64_t j = 0; j < n; ++j) {
                const uint32_t payload = rng.next_u32() & 0x003fffffu;
                switch (rng.below(3)) {
                    case 0: x[j] = f_from_bits(0x7f800000u | payload); break;
                    case 1: x[j] = f_from_bits(0xff800000u | payload); break;
                    default: x[j] = f_from_bits(0x7f800000u | 0x00400000u | payload); break;
                }
            }
            break;
        case Corpus::MixedNonfinite:
            for (int64_t j = 0; j < n; ++j) {
                switch (rng.below(4)) {
                    case 0: x[j] = rng.next_sym(); break;
                    case 1: x[j] = INFINITY; break;
                    case 2: x[j] = -INFINITY; break;
                    default: x[j] = f_from_bits(0xffc00000u | (rng.next_u32() & 0x000fffffu)); break;
                }
            }
            break;
        case Corpus::NanMix:
            // Finite values with a finite maximum, plus a few NaNs scattered in.
            // This is the case the "nan" and "mixed_nonfinite" corpora do not
            // reach: the block maximum stays finite, so an implementation that
            // keeps its fast path here has to reproduce the reference's payload
            // dependent nearest_int(NaN) result for the NaN elements. libFuzzer
            // found exactly this shape (a mutated clamp_edge block with six NaNs
            // added) after 53 executions; see notes/verification.md.
            for (int64_t j = 0; j < n; ++j) {
                x[j] = rng.next_sym();
            }
            for (int64_t k = 0; k < n/32 + 1; ++k) {
                const int64_t  at   = (int64_t) rng.below((uint32_t) n);
                const uint32_t mant = (rng.next_u32() & 0x007fffffu) | 1u;
                const uint32_t sign = rng.next_u32() & 0x80000000u;
                x[at] = f_from_bits(sign | 0x7f800000u | mant);
            }
            break;
        case Corpus::ClampEdge:
            // x[0] = -127 makes max = -127 and iscale = 1.0f exactly, so every
            // other value reaches nearest_int unchanged and the +-k.5 values
            // exercise round-half-to-even; +-127 exercise the MIN(127, .) edge
            for (int64_t j = 0; j < n; ++j) {
                switch (j % 8) {
                    case 0: x[j] = -127.0f; break;
                    case 1: x[j] =  127.0f; break;
                    case 2: x[j] =  (float) rng.below(127) + 0.5f; break;
                    case 3: x[j] = -((float) rng.below(127) + 0.5f); break;
                    case 4: x[j] = -126.9999f; break;
                    case 5: x[j] =  126.9999f; break;
                    case 6: x[j] =  0.5f; break;
                    default: x[j] = -0.5f; break;
                }
            }
            break;
        case Corpus::RandomMix: {
            // one corpus per block, chosen deterministically from the stream
            static const Corpus mix[] = {
                Corpus::Uniform, Corpus::Wide, Corpus::Denormal, Corpus::Tie,
                Corpus::Spike, Corpus::Zero, Corpus::AllZero, Corpus::Extreme,
                Corpus::Inf, Corpus::Nan, Corpus::MixedNonfinite, Corpus::ClampEdge,
            };
            const Corpus pick = mix[rng.below((uint32_t) (sizeof(mix) / sizeof(mix[0])))];
            corpus_fill_block(pick, x, n, rng);
            break;
        }
        default:
            for (int64_t j = 0; j < n; ++j) {
                x[j] = rng.next_sym();
            }
            break;
    }
}

// Fill k values, applying the per-block rule to every 256 block.
static void corpus_fill(Corpus c, float * x, int64_t k, Rng & rng) {
    for (int64_t off = 0; off < k; off += QK_K) {
        const int64_t n = (k - off < QK_K) ? (k - off) : QK_K;
        corpus_fill_block(c, x + off, n, rng);
    }
}
