#!/bin/sh
# Build and run the libFuzzer harness for quantize_row_q8_K.
#
# The CPU backend is rebuilt with -fsanitize=fuzzer-no-link so that coverage is
# attributed to the kernel's own translation unit, plus ASan for memory safety.
# The harness itself provides the fuzzer and ASan runtimes.
#
# Usage: notes/run_fuzz.sh [SECONDS]        (default 900)
# Environment: BIN to point at a different uninstrumented build for the seed
#              corpus generator (default build/release/bin).

set -eu

SECONDS_TOTAL=${1:-900}
BIN=${BIN:-build/release/bin}

CL=$HOME/opt/clang-21/bin
CM=$HOME/.local/bin/cmake

CC=$CL/clang CXX=$CL/clang++ $CM -B build/fuzz -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_FLAGS="-fsanitize=fuzzer-no-link" \
    -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link" \
    -DGGML_SANITIZE_ADDRESS=ON \
    -DGGML_NATIVE=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
    > build/logs/fuzz_configure.log 2>&1

$CM --build build/fuzz -j 10 --target ggml ggml-base > build/logs/fuzz_build.log 2>&1
echo "backend build ok"

$CL/clang++ -std=c++17 -O1 -g -fsanitize=fuzzer,address \
    -I ggml/include -I ggml/src -I ggml/src/ggml-cpu \
    tools/simd-experiment/fuzz-q8k.cpp \
    -L build/fuzz/bin -lggml-cpu -lggml-base -lggml -Wl,-rpath,"$PWD/build/fuzz/bin" \
    -o build/fuzz/fuzz-q8k
echo "harness build ok"

mkdir -p build/fuzz/corpus
"$BIN/simd-diff" --emit-corpus build/fuzz/corpus > /dev/null
# archived inputs that previously crashed the harness, so the regression is
# always part of the corpus even if the generated corpora are regenerated
cp -f notes/fuzz-seeds/*.bin build/fuzz/corpus/ 2>/dev/null || true
echo "seed corpus: $(ls build/fuzz/corpus | wc -l) files, $(du -sh build/fuzz/corpus | cut -f1)"

build/fuzz/fuzz-q8k -max_total_time="$SECONDS_TOTAL" -max_len=8192 -rss_limit_mb=2048 \
    -print_final_stats=1 build/fuzz/corpus 2>&1 | tail -45
