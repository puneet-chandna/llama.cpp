// Bit-exactness differential tester for quantize_row_q8_K.
//
// Compares the CPU backend's quantize_row_q8_K (the "candidate") against
// quantize_row_q8_K_ref (the scalar reference) byte for byte over adversarial
// input corpora. Tolerance is zero: d is a float compared as raw bits, qs is
// 256 int8 values, bsums is 16 int16 values. Any differing byte fails.
//
// Sentinel bytes before and after the output region are compared too, so a
// write outside [0, nb*QK_K) blocks is detected.
//
// Exit status: 0 if every selected corpus is bit-exact, 1 on mismatch, 2 on a
// usage error.
//
// Examples:
//   simd-diff --corpus all --blocks 1000000 --seed 1
//   simd-diff --corpus tie --blocks 100000 --impl candidate
//   simd-diff --self-test
//   simd-diff --q8-0-baseline --blocks 100000        # existing AVX2 q8_0 vs its ref
//   simd-diff --emit-corpus /tmp/seeds               # libFuzzer seed corpus
//   simd-diff --corpus unaligned --misalign 4        # under -fsanitize=undefined

#include "corpus.h"

#include "quants.h"      // quantize_row_q8_K, quantize_row_q8_0
#include "ggml-quants.h" // quantize_row_q8_K_ref, quantize_row_q8_0_ref

#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef void (*q8k_fn)(const float *, void *, int64_t);

static void q8k_ref(const float * x, void * y, int64_t k) {
    quantize_row_q8_K_ref(x, (block_q8_K *) y, k);
}

static void q8k_candidate(const float * x, void * y, int64_t k) {
    quantize_row_q8_K(x, y, k);
}

static void q80_ref(const float * x, void * y, int64_t k) {
    quantize_row_q8_0_ref(x, (block_q8_0 *) y, k);
}

static void q80_candidate(const float * x, void * y, int64_t k) {
    quantize_row_q8_0(x, y, k);
}

// field classification of a byte offset inside a 292 byte block_q8_K
static const char * field_of(size_t off, size_t * index) {
    if (off < 4) {
        *index = 0;
        return "d";
    }
    if (off < 4 + QK_K) {
        *index = off - 4;
        return "qs";
    }
    *index = (off - 4 - QK_K) / sizeof(int16_t);
    return "bsums";
}

// Runs one corpus. Returns the number of calls that differed. Prints the first
// mismatch of the corpus in full detail.
static int64_t run_corpus(const corpus_entry & ce,
                          const std::vector<int64_t> & ks,
                          bool unaligned,
                          size_t misalign_step,
                          int64_t calls,
                          uint64_t seed,
                          q8k_fn fn_a,
                          q8k_fn fn_b,
                          bool verbose_first,
                          int64_t * blocks_out) {
    const Corpus corpus = ce.id;

    int64_t k_max = 0;
    for (int64_t k : ks) {
        if (k > k_max) {
            k_max = k;
        }
    }
    const size_t nb_max  = (size_t) (k_max / QK_K);
    const size_t out_max = nb_max * sizeof(block_q8_K);

    const size_t pad = 64;
    std::vector<float>   xgen((size_t) k_max + 1);
    std::vector<uint8_t> xbuf((size_t) k_max * sizeof(float) + pad + 32);
    std::vector<uint8_t> yaout(out_max + pad + 32);
    std::vector<uint8_t> ybout(out_max + pad + 32);

    Rng rng(seed);

    int64_t mismatches = 0;
    bool printed = false;

    if (blocks_out != nullptr) {
        *blocks_out = 0;
    }

    for (int64_t call = 0; call < calls; ++call) {
        const int64_t k  = ks[(size_t) (call % (int64_t) ks.size())];
        const size_t  nb = (size_t) (k / QK_K);

        if (blocks_out != nullptr) {
            *blocks_out += (int64_t) nb;
        }

        const size_t nstep = (misalign_step > 32) ? 32 : misalign_step;
        const size_t ox = unaligned ? nstep * (size_t) rng.below((uint32_t) (32 / nstep)) : 0;
        const size_t oy = unaligned ? nstep * (size_t) rng.below((uint32_t) (32 / nstep)) : 0;

        corpus_fill(corpus, xgen.data(), k, rng);
        std::memset(xbuf.data(), 0x00, xbuf.size());
        if (k > 0) {
            std::memcpy(xbuf.data() + ox, xgen.data(), (size_t) k * sizeof(float));
        }

        std::memset(yaout.data(), 0x5a, yaout.size());
        std::memset(ybout.data(), 0x5a, ybout.size());

        const size_t out_bytes = nb * sizeof(block_q8_K);
        uint8_t * const ya = yaout.data() + oy;
        uint8_t * const yb = ybout.data() + oy;

        fn_a((const float *) (void *) (xbuf.data() + ox), ya, k);
        fn_b((const float *) (void *) (xbuf.data() + ox), yb, k);

        const size_t cmp_bytes = oy + out_bytes + 32;
        if (std::memcmp(yaout.data(), ybout.data(), cmp_bytes) == 0) {
            continue;
        }

        ++mismatches;
        if (printed || !verbose_first) {
            continue;
        }
        printed = true;

        size_t off = 0;
        while (off < cmp_bytes && yaout.data()[off] == ybout.data()[off]) {
            ++off;
        }
        printf("MISMATCH corpus=%s call=%" PRId64 " k=%" PRId64 " blocks=%zu x_off=%zu y_off=%zu byte=%zu\n",
               ce.name, call, k, nb, ox, oy, off);

        if (off >= oy && off < oy + out_bytes) {
            const size_t boff = off - oy;
            const size_t blk  = boff / sizeof(block_q8_K);
            const size_t foff = boff % sizeof(block_q8_K);
            size_t fidx = 0;
            const char * fname = field_of(foff, &fidx);
            if (std::strcmp(fname, "d") == 0) {
                uint32_t ra = 0;
                uint32_t rb = 0;
                std::memcpy(&ra, ya + blk * sizeof(block_q8_K), 4);
                std::memcpy(&rb, yb + blk * sizeof(block_q8_K), 4);
                printf("  region=block_%zu_d ref=0x%08" PRIx32 " candidate=0x%08" PRIx32 "\n", blk, ra, rb);
            } else if (std::strcmp(fname, "qs") == 0) {
                printf("  region=block_%zu_qs[%zu] ref=%d candidate=%d\n",
                       blk, fidx, (int) (int8_t) ya[boff], (int) (int8_t) yb[boff]);
            } else {
                int16_t ra = 0;
                int16_t rb = 0;
                std::memcpy(&ra, ya + blk * sizeof(block_q8_K) + foff, 2);
                std::memcpy(&rb, yb + blk * sizeof(block_q8_K) + foff, 2);
                printf("  region=block_%zu_bsums[%zu] ref=%d candidate=%d\n", blk, fidx, (int) ra, (int) rb);
            }
            printf("  inputs of block %zu (256 float bit patterns, 8 per line):\n", blk);
            for (int64_t j = 0; j < QK_K; ++j) {
                uint32_t bits = 0;
                std::memcpy(&bits, xbuf.data() + ox + (size_t) blk * QK_K * 4 + (size_t) j * 4, 4);
                printf("    %08" PRIx32 "%s", bits, (j % 8 == 7) ? "\n" : "");
            }
        } else {
            printf("  region=%s byte=%zu ref=0x%02x candidate=0x%02x\n",
                   (off < oy) ? "write_before_output" : "write_after_output",
                   off, yaout.data()[off], ybout.data()[off]);
        }
    }

    return mismatches;
}

static void emit_seed_corpus(const char * dir, uint64_t seed) {
    // seed file format, mirrored by fuzz-q8k.cpp:
    //   byte 0: k_blocks = value % 6   -> k = 256*k_blocks + k_tail
    //   byte 1: k_tail   = value % 256
    //   byte 2: x byte offset
    //   byte 3: y byte offset
    //   bytes 4..: little endian float payload, 1536 floats
    const int64_t n_floats = 1536;
    std::vector<float> x((size_t) n_floats);
    for (int ci = 0; ci < corpus_count(); ++ci) {
        const corpus_entry & ce = CORPUS_TABLE[ci];
        const struct { const char * tag; uint8_t hdr[4]; } variants[2] = {
            { "aligned", { 4, 0, 0,  0 } },
            { "tail",    { 4, 4, 3, 12 } },
        };
        for (const auto & v : variants) {
            Rng rng(seed);
            corpus_fill(ce.id, x.data(), n_floats, rng);
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s_%s.bin", dir, ce.name, v.tag);
            FILE * f = fopen(path, "wb");
            if (f == nullptr) {
                fprintf(stderr, "error: cannot write %s\n", path);
                exit(2);
            }
            fwrite(v.hdr, 1, 4, f);
            fwrite(x.data(), sizeof(float), (size_t) n_floats, f);
            fclose(f);
            printf("wrote %s\n", path);
        }
    }
}

static void usage(const char * argv0) {
    printf("usage: %s [options]\n", argv0);
    printf("  --corpus NAME|all     corpus selection (default all)\n");
    printf("  --blocks N            number of 256-element blocks per corpus (default 100000)\n");
    printf("  --calls N             number of calls per corpus, overrides --blocks\n");
    printf("  --k LIST              comma separated element counts (default 256)\n");
    printf("  --seed S              input seed (default 1)\n");
    printf("  --impl candidate|ref  second implementation; ref compares the reference against itself\n");
    printf("  --misalign STEP       byte offsets are multiples of STEP up to 31 (default 1)\n");
    printf("  --self-test           inject a one byte difference and require it to be reported\n");
    printf("  --q8-0-baseline       compare quantize_row_q8_0 (existing AVX2) against its reference\n");
    printf("  --emit-corpus DIR     write the libFuzzer seed corpus and exit\n");
    printf("  --quiet               only print the per corpus summary\n");
    printf("\ncorpora:");
    for (int i = 0; i < corpus_count(); ++i) {
        printf(" %s", CORPUS_TABLE[i].name);
    }
    printf("\n");
}

// q8_0 baseline driver: same block accounting as the q8_K driver, one call is
// 8 q8_0 blocks, so the element count matches a 256 element q8_K call.
static int64_t run_q80_baseline(const corpus_entry & ce, int64_t calls, uint64_t seed) {
    const int64_t k0 = 256;
    std::vector<float>   x((size_t) k0);
    std::vector<uint8_t> ya((size_t) k0 / 32 * sizeof(block_q8_0) + 96);
    std::vector<uint8_t> yb((size_t) k0 / 32 * sizeof(block_q8_0) + 96);
    const size_t out_bytes = (size_t) k0 / 32 * sizeof(block_q8_0);

    Rng rng(seed);
    int64_t mismatches = 0;
    for (int64_t call = 0; call < calls; ++call) {
        corpus_fill(ce.id, x.data(), k0, rng);
        std::memset(ya.data(), 0x5a, ya.size());
        std::memset(yb.data(), 0x5a, yb.size());
        q80_ref(x.data(), ya.data() + 32, k0);
        q80_candidate(x.data(), yb.data() + 32, k0);
        if (std::memcmp(ya.data(), yb.data(), 32 + out_bytes + 32) != 0) {
            ++mismatches;
            if (mismatches == 1) {
                size_t off = 0;
                while (off < 32 + out_bytes + 32 && ya.data()[off] == yb.data()[off]) {
                    ++off;
                }
                printf("  q8_0 first differing byte %zu (block %zu): ref=0x%02x candidate=0x%02x\n",
                       off, (off >= 32) ? (off - 32) / sizeof(block_q8_0) : 0,
                       ya.data()[off], yb.data()[off]);
            }
        }
    }
    return mismatches;
}

int main(int argc, char ** argv) {
    std::string corpus_sel = "all";
    int64_t blocks = 100000;
    int64_t calls  = -1;
    uint64_t seed  = 1;
    std::string impl = "candidate";
    size_t misalign_step = 1;
    bool self_test = false;
    bool q80_baseline = false;
    bool quiet = false;
    std::string emit_dir;
    std::vector<int64_t> ks;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next_arg = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s needs a value\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--corpus") {
            corpus_sel = next_arg();
        } else if (a == "--blocks") {
            blocks = strtoll(next_arg(), nullptr, 10);
        } else if (a == "--calls") {
            calls = strtoll(next_arg(), nullptr, 10);
        } else if (a == "--seed") {
            seed = strtoull(next_arg(), nullptr, 10);
        } else if (a == "--impl") {
            impl = next_arg();
        } else if (a == "--misalign") {
            misalign_step = (size_t) strtoll(next_arg(), nullptr, 10);
        } else if (a == "--k") {
            std::string list = next_arg();
            size_t pos = 0;
            while (pos <= list.size()) {
                const size_t comma = list.find(',', pos);
                const std::string item = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!item.empty()) {
                    ks.push_back(strtoll(item.c_str(), nullptr, 10));
                }
                if (comma == std::string::npos) {
                    break;
                }
                pos = comma + 1;
            }
        } else if (a == "--self-test") {
            self_test = true;
        } else if (a == "--q8-0-baseline") {
            q80_baseline = true;
        } else if (a == "--emit-corpus") {
            emit_dir = next_arg();
        } else if (a == "--quiet") {
            quiet = true;
        } else if (a == "--help" || a == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown option %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    if (misalign_step < 1) {
        fprintf(stderr, "error: --misalign must be >= 1\n");
        return 2;
    }

    if (!emit_dir.empty()) {
        emit_seed_corpus(emit_dir.c_str(), seed);
        return 0;
    }

    if (impl != "candidate" && impl != "ref") {
        fprintf(stderr, "error: --impl must be candidate or ref\n");
        return 2;
    }

    q8k_fn fn_a = q8k_ref;
    q8k_fn fn_b = (impl == "ref") ? q8k_ref : q8k_candidate;

    if (self_test) {
        // Prove that the comparison actually compares: inject one byte into the
        // second output buffer and require the same comparison to report it.
        const int64_t k = 256;
        std::vector<float>   x((size_t) k);
        std::vector<uint8_t> ya(sizeof(block_q8_K) + 96);
        std::vector<uint8_t> yb(sizeof(block_q8_K) + 96);
        Rng rng(seed);
        corpus_fill(Corpus::Uniform, x.data(), k, rng);
        std::memset(ya.data(), 0x5a, ya.size());
        std::memset(yb.data(), 0x5a, yb.size());
        fn_a(x.data(), ya.data() + 32, k);
        fn_b(x.data(), yb.data() + 32, k);
        const size_t cmp = 32 + sizeof(block_q8_K) + 32;
        const bool clean = std::memcmp(ya.data(), yb.data(), cmp) == 0;
        yb.data()[32 + 36 + 100] ^= 0xff; // inside qs
        const bool detected = std::memcmp(ya.data(), yb.data(), cmp) != 0;
        yb.data()[32 + 36 + 100] ^= 0xff;
        const bool clean_again = std::memcmp(ya.data(), yb.data(), cmp) == 0;
        printf("self-test: %s (clean=%s injected=%s reverted=%s)\n",
               (clean && detected && clean_again) ? "PASS" : "FAIL",
               clean ? "clean" : "dirty",
               detected ? "detected" : "NOT detected",
               clean_again ? "clean" : "dirty");
        return (clean && detected && clean_again) ? 0 : 1;
    }

    std::vector<corpus_entry> selected;
    if (corpus_sel == "all") {
        for (int i = 0; i < corpus_count(); ++i) {
            selected.push_back(CORPUS_TABLE[i]);
        }
    } else {
        bool found = false;
        for (int i = 0; i < corpus_count(); ++i) {
            if (corpus_sel == CORPUS_TABLE[i].name) {
                selected.push_back(CORPUS_TABLE[i]);
                found = true;
            }
        }
        if (!found) {
            fprintf(stderr, "error: unknown corpus %s\n", corpus_sel.c_str());
            return 2;
        }
    }

    if (!quiet) {
        printf("simd-diff impl=%s blocks=%" PRId64 " seed=%" PRIu64 " misalign_step=%zu\n",
               impl.c_str(), blocks, seed, misalign_step);
    }

    int64_t total_mismatches = 0;
    size_t seed_i = 0;
    for (const corpus_entry & ce : selected) {
        std::vector<int64_t> corpus_ks = ks;

        if (q80_baseline) {
            const int64_t blocks_per_call = 8; // 256 elements per call
            int64_t n_calls = calls;
            if (n_calls < 0) {
                n_calls = blocks / blocks_per_call;
            }
            if (n_calls < 1) {
                n_calls = 1;
            }
            const int64_t mismatches = run_q80_baseline(ce, n_calls, seed + 1000 + seed_i);
            printf("q8_0_baseline corpus=%-15s calls=%-9" PRId64 " blocks=%-11" PRId64 " mismatches=%" PRId64 "\n",
                   ce.name, n_calls, n_calls * blocks_per_call, mismatches);
            total_mismatches += mismatches;
            ++seed_i;
            continue;
        }

        bool unaligned = false;
        if (corpus_ks.empty()) {
            if (ce.id == Corpus::OddK) {
#if defined(NDEBUG)
                // k values that are not multiples of QK_K: both implementations
                // must ignore the tail identically. Requires NDEBUG because the
                // reference asserts k % QK_K == 0.
                corpus_ks = { 0, 256, 640, 2303, 769 };
#else
                printf("corpus=%-15s SKIPPED (asserts enabled here, k %% 256 != 0 would abort)\n", ce.name);
                ++seed_i;
                continue;
#endif
            } else {
                corpus_ks = { 256 };
            }
        }
        if (ce.id == Corpus::Unaligned) {
            unaligned = true;
        }

        int64_t smallest_k = corpus_ks[0];
        for (int64_t k : corpus_ks) {
            if (k < smallest_k) {
                smallest_k = k;
            }
        }
        const int64_t blocks_per_call = (smallest_k > 0) ? (smallest_k / QK_K) : 1;
        int64_t n_calls = calls;
        if (n_calls < 0) {
            n_calls = blocks / (blocks_per_call > 0 ? blocks_per_call : 1);
        }
        if (n_calls < 1) {
            n_calls = 1;
        }

        int64_t blocks_cmp = 0;
        const int64_t mismatches = run_corpus(ce, corpus_ks, unaligned, misalign_step,
                                              n_calls, seed + 1000 + seed_i, fn_a, fn_b, !quiet,
                                              &blocks_cmp);
        printf("corpus=%-15s calls=%-9" PRId64 " blocks=%-11" PRId64 " mismatches=%" PRId64 "\n",
               ce.name, n_calls, blocks_cmp, mismatches);

        total_mismatches += mismatches;
        ++seed_i;
    }

    printf("total mismatches: %" PRId64 "\n", total_mismatches);
    return total_mismatches == 0 ? 0 : 1;
}
