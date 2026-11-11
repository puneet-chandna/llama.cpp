#!/bin/sh
# llama-bench only, with weight repacking disabled, so that every K/IQ tensor
# goes through quantize_row_q8_K. Used unchanged for the phase 3 baseline and
# the phase 5 re-run.
#
# Note: llama-bench does not accept the common `-nr` / `--no-repack` flag
# (common/arg.cpp:2395 belongs to the gpt_params parser); it has its own
# `--repack <0|1>` option (tools/llama-bench/llama-bench.cpp:915) which sets
# llama_model_params.use_extra_bufts.
#
# Usage: notes/run_bench_norepack.sh OUTPUT.txt LABEL
# Environment: BIN overrides the build directory (default build/release/bin).

set -eu

OUT=${1:?usage: run_bench_norepack.sh OUTPUT.txt LABEL}
LABEL=${2:?usage: run_bench_norepack.sh OUTPUT.txt LABEL}
BIN=${BIN:-build/release/bin}

PIN="taskset -c 0,2"
THREADS=2

mkdir -p build/logs

mem_avail_mb() {
    awk '/MemAvailable/ { printf "%d", $2/1024 }' /proc/meminfo
}

{
    echo "################################################################"
    echo "# label:    $LABEL (repack disabled)"
    echo "# date:     $(date -Is)"
    echo "# bin:      $BIN"
    echo "# measured: sha256($BIN/libggml-cpu.so) = $(sha256sum "$BIN/libggml-cpu.so" | cut -c1-32)"
    echo "# governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    echo "# mem avail: $(mem_avail_mb) MB"
    echo "# config:   $PIN, -t $THREADS, --repack 0"
    echo "################################################################"

    # name:model:pp:tg:reps:min_mem_mb
    SPECS="1B:models/Llama-3.2-1B-Instruct-Q4_K_M.gguf:512:128:5:2048
3B:models/Llama-3.2-3B-Instruct-IQ4_XS.gguf:512:128:5:3000
7B:models/Qwen2.5-7B-Instruct-Q4_K_M.gguf:256:64:3:5200"

    for spec in $SPECS; do
        name=$(echo "$spec"  | cut -d: -f1)
        model=$(echo "$spec" | cut -d: -f2)
        pp=$(echo "$spec"    | cut -d: -f3)
        tg=$(echo "$spec"    | cut -d: -f4)
        reps=$(echo "$spec"  | cut -d: -f5)
        need=$(echo "$spec"  | cut -d: -f6)

        avail=$(mem_avail_mb)
        if [ "$avail" -lt "$need" ]; then
            echo
            echo "=== $name SKIPPED: MemAvailable ${avail} MB < ${need} MB"
            continue
        fi

        echo
        echo "=== llama-bench $name norepack (mem avail ${avail} MB) ==="
        echo "# $PIN $BIN/llama-bench -m $model -p $pp -n $tg -t $THREADS -r $reps --repack 0 -o md"
        $PIN "$BIN/llama-bench" -m "$model" -p "$pp" -n "$tg" -t "$THREADS" -r "$reps" --repack 0 -o md \
            || echo "llama-bench FAILED with status $?"
    done

    echo
    echo "=== mem avail at end: $(mem_avail_mb) MB ==="
    echo "=== done ==="
} > "$OUT" 2>&1

echo "wrote $OUT"
