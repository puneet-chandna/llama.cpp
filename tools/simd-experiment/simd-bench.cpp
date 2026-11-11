// Microbenchmark for quantize_row_q8_K (candidate) against quantize_row_q8_K_ref.
//
// "candidate" resolves to the symbol the CPU backend actually uses, which is
// quantize_row_q8_K from arch/x86/quants.c. The tool does not need rebuilding
// when that implementation changes, so baseline and post-change numbers are
// directly comparable.
//
// Modes:
//   single  one call of k elements per timed sample, repeated
//   rows    R calls of k elements per timed sample (the activation packing
//           pattern of a matmul with R src1 rows)
//
// Examples:
//   simd-bench --impl both --k 256,4096,32768 --reps 21
//   simd-bench --impl both --mode rows --rows 512 --k 2048 --check

#include "corpus.h"

#include "quants.h"      // quantize_row_q8_K
#include "ggml-quants.h" // quantize_row_q8_K_ref

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

typedef void (*q8k_fn)(const float *, void *, int64_t);

static void q8k_ref(const float * x, void * y, int64_t k) {
    quantize_row_q8_K_ref(x, (block_q8_K *) y, k);
}

static void q8k_candidate(const float * x, void * y, int64_t k) {
    quantize_row_q8_K(x, y, k);
}

static const char * impl_name(q8k_fn fn) {
    return fn == q8k_ref ? "ref" : "candidate";
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

struct stats {
    double median = 0.0;
    double mean   = 0.0;
    double sd     = 0.0;
    double min    = 0.0;
    size_t n      = 0;
};

static stats summarize(std::vector<double> v) {
    stats s;
    s.n = v.size();
    if (v.empty()) {
        return s;
    }
    std::sort(v.begin(), v.end());
    s.min    = v.front();
    s.median = (v.size() % 2) ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    double sum = 0.0;
    for (double x : v) {
        sum += x;
    }
    s.mean = sum / (double) v.size();
    double acc = 0.0;
    for (double x : v) {
        acc += (x - s.mean) * (x - s.mean);
    }
    s.sd = (v.size() > 1) ? sqrt(acc / (double) (v.size() - 1)) : 0.0;
    return s;
}

static void run_rows(q8k_fn fn, const float * x, void * y, int64_t k, int64_t rows) {
    const size_t bs = (size_t) (k / QK_K) * sizeof(block_q8_K);
    for (int64_t r = 0; r < rows; ++r) {
        fn(x + r * k, (char *) y + (size_t) r * bs, k);
    }
}

static void usage(const char * argv0) {
    printf("usage: %s [options]\n", argv0);
    printf("  --impl candidate|ref|both   implementations to time (default both)\n");
    printf("  --k LIST                    comma separated element counts (default 256,512,2048,4096,8192,32768)\n");
    printf("  --mode single|rows          timing mode (default single)\n");
    printf("  --rows R                    rows per timed sample in rows mode (default 1)\n");
    printf("  --reps N                    timed samples (default 21)\n");
    printf("  --seed S                    input seed (default 12345)\n");
    printf("  --check                     bit-compare candidate against ref on the timed buffers\n");
    printf("  --csv                       print results as CSV\n");
}

int main(int argc, char ** argv) {
    std::vector<int64_t> ks = { 256, 512, 2048, 4096, 8192, 32768 };
    std::string impl = "both";
    std::string mode = "single";
    int64_t rows = 1;
    int64_t reps = 21;
    uint64_t seed = 12345;
    bool check = false;
    bool csv = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next_arg = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s needs a value\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--impl") {
            impl = next_arg();
        } else if (a == "--mode") {
            mode = next_arg();
        } else if (a == "--rows") {
            rows = strtoll(next_arg(), nullptr, 10);
        } else if (a == "--reps") {
            reps = strtoll(next_arg(), nullptr, 10);
        } else if (a == "--seed") {
            seed = strtoull(next_arg(), nullptr, 10);
        } else if (a == "--k") {
            ks.clear();
            std::string list = next_arg();
            size_t pos = 0;
            while (pos <= list.size()) {
                const size_t comma = list.find(',', pos);
                const std::string item = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!item.empty()) {
                    const long long v = strtoll(item.c_str(), nullptr, 10);
                    if (v < 0 || v % QK_K != 0) {
                        fprintf(stderr, "error: k must be a non-negative multiple of %d, got %lld\n", QK_K, v);
                        return 2;
                    }
                    ks.push_back(v);
                }
                if (comma == std::string::npos) {
                    break;
                }
                pos = comma + 1;
            }
        } else if (a == "--check") {
            check = true;
        } else if (a == "--csv") {
            csv = true;
        } else if (a == "--help" || a == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown option %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    if (mode != "single" && mode != "rows") {
        fprintf(stderr, "error: --mode must be single or rows\n");
        return 2;
    }
    if (mode == "single") {
        rows = 1;
    }
    if (rows < 1 || reps < 1) {
        fprintf(stderr, "error: --rows and --reps must be >= 1\n");
        return 2;
    }

    std::vector<q8k_fn> impls;
    if (impl == "both") {
        impls = { q8k_ref, q8k_candidate };
    } else if (impl == "ref") {
        impls = { q8k_ref };
    } else if (impl == "candidate") {
        impls = { q8k_candidate };
    } else {
        fprintf(stderr, "error: --impl must be candidate, ref or both\n");
        return 2;
    }

    Rng rng(seed);

    if (!csv) {
        printf("impl=%s mode=%s rows=%" PRId64 " reps=%" PRId64 " warmup=3 seed=%" PRIu64 "\n",
               impl.c_str(), mode.c_str(), rows, reps, seed);
    } else {
        printf("impl,k,rows,median_ns,ns_per_elem,gbps,min_ns,sd_ns\n");
    }

    // samples[impl_index][k_index][rep]; the two implementations are timed
    // alternately inside each repetition so that the ratio between them is
    // measured under the same clock and thermal state. Absolute times on this
    // host drift by more than a factor of one between runs.
    std::vector<std::vector<std::vector<double>>> samples(impls.size());
    for (auto & v : samples) {
        v.resize(ks.size());
    }

    for (size_t ki = 0; ki < ks.size(); ++ki) {
        const int64_t k  = ks[ki];
        const size_t  nb = (size_t) (k / QK_K);

        std::vector<float> x((size_t) k * (size_t) rows);
        std::vector<block_q8_K> y(nb * (size_t) rows);

        for (int64_t r = 0; r < rows; ++r) {
            corpus_fill(Corpus::Uniform, x.data() + (size_t) r * (size_t) k, k, rng);
        }
        std::memset(y.data(), 0, y.size() * sizeof(block_q8_K));

        for (size_t ii = 0; ii < impls.size(); ++ii) {
            for (int w = 0; w < 3; ++w) {
                run_rows(impls[ii], x.data(), y.data(), k, rows);
            }
        }

        for (size_t ii = 0; ii < impls.size(); ++ii) {
            samples[ii][ki].reserve((size_t) reps);
        }
        for (int64_t rep = 0; rep < reps; ++rep) {
            for (size_t ii = 0; ii < impls.size(); ++ii) {
                const uint64_t t0 = now_ns();
                run_rows(impls[ii], x.data(), y.data(), k, rows);
                const uint64_t t1 = now_ns();
                samples[ii][ki].push_back((double) (t1 - t0) / (double) rows);
            }
        }
    }

    std::vector<std::vector<stats>> results(impls.size());
    for (size_t ii = 0; ii < impls.size(); ++ii) {
        results[ii].resize(ks.size());
        for (size_t ki = 0; ki < ks.size(); ++ki) {
            results[ii][ki] = summarize(samples[ii][ki]);
        }
    }

    for (size_t ii = 0; ii < impls.size(); ++ii) {
        for (size_t ki = 0; ki < ks.size(); ++ki) {
            const int64_t k   = ks[ki];
            const stats & s   = results[ii][ki];
            const double bytes = 4.0 * (double) k + (double) (k / QK_K) * (double) sizeof(block_q8_K);
            const double gbps  = (s.median > 0.0) ? bytes / s.median : 0.0;
            const double per_e = (k > 0) ? s.median / (double) k : 0.0;
            if (csv) {
                printf("%s,%" PRId64 ",%" PRId64 ",%.1f,%.4f,%.2f,%.1f,%.1f\n",
                       impl_name(impls[ii]), k, rows, s.median, per_e, gbps, s.min, s.sd);
            } else {
                printf("%-9s k=%-7" PRId64 " rows=%-5" PRId64 " median=%10.1f ns  ns/elem=%8.4f  GB/s=%7.2f  min=%10.1f  sd=%8.1f\n",
                       impl_name(impls[ii]), k, rows, s.median, per_e, gbps, s.min, s.sd);
            }
        }
    }

    if (impls.size() == 2 && !csv) {
        printf("\nspeedup (ref / candidate), paired per repetition:\n");
        for (size_t ki = 0; ki < ks.size(); ++ki) {
            const double r = results[0][ki].median;
            const double c = results[1][ki].median;
            // per repetition ratios, so that clock drift between the two
            // implementations cancels instead of biasing the result
            std::vector<double> ratios;
            const size_t n = samples[0][ki].size() < samples[1][ki].size() ? samples[0][ki].size() : samples[1][ki].size();
            for (size_t i = 0; i < n; ++i) {
                if (samples[1][ki][i] > 0.0) {
                    ratios.push_back(samples[0][ki][i] / samples[1][ki][i]);
                }
            }
            const stats rs = summarize(ratios);
            printf("  k=%-7" PRId64 " ref=%10.1f ns  candidate=%10.1f ns  ratio(median)=%6.3fx  ratio(sd)=%5.3f\n",
                   ks[ki], r, c, rs.median, rs.sd);
        }
    }

    int rc = 0;
    if (check) {
        // independent buffers, bit compare over the whole output of every k
        for (size_t ki = 0; ki < ks.size(); ++ki) {
            const int64_t k  = ks[ki];
            const size_t  nb = (size_t) (k / QK_K);
            std::vector<float> x((size_t) k * (size_t) rows);
            std::vector<block_q8_K> ya(nb * (size_t) rows);
            std::vector<block_q8_K> yb(nb * (size_t) rows);
            for (int64_t r = 0; r < rows; ++r) {
                corpus_fill(Corpus::Uniform, x.data() + (size_t) r * (size_t) k, k, rng);
            }
            std::memset(ya.data(), 0xa5, ya.size() * sizeof(block_q8_K));
            std::memset(yb.data(), 0xa5, yb.size() * sizeof(block_q8_K));
            run_rows(q8k_ref,       x.data(), ya.data(), k, rows);
            run_rows(q8k_candidate, x.data(), yb.data(), k, rows);

            const size_t bytes = nb * (size_t) rows * sizeof(block_q8_K);
            if (std::memcmp(ya.data(), yb.data(), bytes) != 0) {
                printf("check: MISMATCH k=%" PRId64 " rows=%" PRId64 "\n", k, rows);
                const uint8_t * pa = (const uint8_t *) ya.data();
                const uint8_t * pb = (const uint8_t *) yb.data();
                for (size_t b = 0; b < bytes; ++b) {
                    if (pa[b] != pb[b]) {
                        printf("  first differing byte %zu: ref=0x%02x candidate=0x%02x (block %zu, field offset %zu)\n",
                               b, pa[b], pb[b], b / sizeof(block_q8_K), b % sizeof(block_q8_K));
                        break;
                    }
                }
                rc = 1;
            } else {
                printf("check: OK k=%" PRId64 " rows=%" PRId64 " bytes=%zu\n", k, rows, bytes);
            }
        }
    }

    return rc;
}
