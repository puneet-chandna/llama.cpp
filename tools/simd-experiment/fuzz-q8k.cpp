// libFuzzer + ASan/UBSan harness for quantize_row_q8_K.
//
// Input format (the same one `simd-diff --emit-corpus` writes):
//   byte 0: k_blocks = value % 6   -> k = 256*k_blocks + k_tail
//   byte 1: k_tail   = value % 256
//   byte 2: x byte offset
//   byte 3: y byte offset
//   bytes 4..: little endian float payload, up to 1536 floats
//
// The harness aborts on any byte difference between the CPU backend's
// quantize_row_q8_K and quantize_row_q8_K_ref.
//
// The input and output allocations are sized to end exactly at the last byte the
// implementations are allowed to touch, so ASan turns any read past k elements
// or any write past the last block into a hard failure rather than a silent
// difference. Because the reference asserts k % QK_K == 0, the harness calls
// with k rounded down to a multiple of QK_K; the "ignores the tail" property is
// covered by `simd-diff --corpus odd_k` in builds without asserts (NDEBUG).
//
// Build (from the repository root):
//   clang++ -std=c++17 -O1 -g -fsanitize=fuzzer,address \
//     -I ggml/include -I ggml/src -I ggml/src/ggml-cpu \
//     tools/simd-experiment/fuzz-q8k.cpp \
//     -L build/fuzz/bin -lggml-cpu -lggml-base -lggml -Wl,-rpath,$PWD/build/fuzz/bin \
//     -o build/fuzz/fuzz-q8k
//   mkdir -p build/fuzz/corpus && build/release/bin/simd-diff --emit-corpus build/fuzz/corpus
//   build/fuzz/fuzz-q8k -max_total_time=900 -max_len=8192 -rss_limit_mb=2048 build/fuzz/corpus

#include "corpus.h"

#include "quants.h"      // quantize_row_q8_K
#include "ggml-quants.h" // quantize_row_q8_K_ref

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    if (size < 8) {
        return 0;
    }

    const int64_t k_requested = 256 * (int64_t) (data[0] % 6) + (int64_t) (data[1] % 256);
    const size_t  x_off       = (size_t) data[2];
    const size_t  y_off       = (size_t) data[3];

    const int64_t k  = k_requested - (k_requested % QK_K); // the reference asserts k % QK_K == 0
    const size_t  nb = (size_t) (k / QK_K);
    if (k <= 0) {
        return 0;
    }

    const size_t in_bytes  = (size_t) k * sizeof(float);
    const size_t out_bytes = nb * sizeof(block_q8_K);

    // payload: as many floats as the input carries, zero padded
    std::vector<uint8_t> xraw(in_bytes + x_off);
    std::memset(xraw.data(), 0, xraw.size());
    const size_t avail = size - 4;
    std::memcpy(xraw.data() + x_off, data + 4, avail < in_bytes ? avail : in_bytes);

    std::vector<uint8_t> ya(out_bytes + y_off);
    std::vector<uint8_t> yb(out_bytes + y_off);
    std::memset(ya.data(), 0x5a, ya.size());
    std::memset(yb.data(), 0x5a, yb.size());

    const float * const x = (const float *) (const void *) (xraw.data() + x_off);
    void * const yref = ya.data() + y_off;
    void * const ycmp = yb.data() + y_off;

    quantize_row_q8_K_ref(x, (block_q8_K *) yref, k);
    quantize_row_q8_K(x, ycmp, k);

    if (std::memcmp(ya.data(), yb.data(), ya.size()) != 0) {
        size_t off = 0;
        while (off < ya.size() && ya.data()[off] == yb.data()[off]) {
            ++off;
        }
        fprintf(stderr,
                "MISMATCH k=%lld nb=%zu x_off=%zu y_off=%zu first_byte=%zu ref=0x%02x candidate=0x%02x\n",
                (long long) k, nb, x_off, y_off, off, ya.data()[off], yb.data()[off]);
        abort();
    }

    return 0;
}
