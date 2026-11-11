#!/bin/sh
# End-to-end A/B benchmark driver, used unchanged for the phase 3 baseline and
# the phase 5 re-run so both sides use identical commands.
#
# Usage: notes/run_bench.sh OUTPUT.txt LABEL
#
# Fixed measurement configuration:
#   taskset -c 0,2   one hardware thread per P core, no SMT sharing
#   -t 2             two compute threads, matching the pinning
#   5 repetitions    (3 for the 7B, which is memory limited on this machine)
#
# Environment: BIN overrides the build directory (default build/release/bin).

set -eu

OUT=${1:?usage: run_bench.sh OUTPUT.txt LABEL}
LABEL=${2:?usage: run_bench.sh OUTPUT.txt LABEL}
BIN=${BIN:-build/release/bin}

PIN="taskset -c 0,2"
THREADS=2

M1B=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
M3B=models/Llama-3.2-3B-Instruct-IQ4_XS.gguf
M7B=models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
WIKI=models/wikitext-2-raw/wiki.test.raw

mkdir -p build/logs

mem_avail_mb() {
    awk '/MemAvailable/ { printf "%d", $2/1024 }' /proc/meminfo
}

{
    echo "################################################################"
    echo "# label:    $LABEL"
    echo "# date:     $(date -Is)"
    echo "# bin:      $BIN"
    echo "# git:      $(git rev-parse --short HEAD), $(git status --porcelain | wc -l) modified files"
    echo "# measured: sha256($BIN/libggml-cpu.so) = $(sha256sum "$BIN/libggml-cpu.so" | cut -c1-32)"
    echo "# cpu:      $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
    echo "# governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    echo "# mem avail: $(mem_avail_mb) MB"
    echo "# config:   $PIN, -t $THREADS"
    echo "################################################################"

    # name:model:pp:tg:bench_reps:ppl_chunks:ppl_reps:min_mem_mb
    SPECS="1B:$M1B:512:128:5:16:5:2048
3B:$M3B:512:128:5:16:5:3000
7B:$M7B:256:64:3:8:3:5200"

    for spec in $SPECS; do
        name=$(echo "$spec"  | cut -d: -f1)
        model=$(echo "$spec" | cut -d: -f2)
        pp=$(echo "$spec"    | cut -d: -f3)
        tg=$(echo "$spec"    | cut -d: -f4)
        reps=$(echo "$spec"  | cut -d: -f5)
        chunks=$(echo "$spec"| cut -d: -f6)
        preps=$(echo "$spec" | cut -d: -f7)
        need=$(echo "$spec"  | cut -d: -f8)

        avail=$(mem_avail_mb)
        if [ "$avail" -lt "$need" ]; then
            echo
            echo "=== $name SKIPPED: MemAvailable ${avail} MB < ${need} MB"
            continue
        fi

        echo
        echo "=== llama-bench $name (mem avail ${avail} MB) ==="
        echo "# $PIN $BIN/llama-bench -m $model -p $pp -n $tg -t $THREADS -r $reps -o md"
        $PIN "$BIN/llama-bench" -m "$model" -p "$pp" -n "$tg" -t "$THREADS" -r "$reps" -o md \
            || echo "llama-bench FAILED with status $?"

        echo
        echo "=== llama-perplexity $name (chunks $chunks, $preps reps) ==="
        i=1
        while [ "$i" -le "$preps" ]; do
            raw=build/logs/ppl_${LABEL}_${name}_${i}.log
            $PIN "$BIN/llama-perplexity" -m "$model" -f "$WIKI" -c 512 --chunks "$chunks" -t "$THREADS" > "$raw" 2>&1 || true
            echo "rep $i chunks: $(grep -o '\[[0-9]*\][0-9.]*' "$raw" | tr '\n' ' ' | cut -c1-200)"
            echo "rep $i final:  $(grep -o 'Final estimate:.*' "$raw" | tail -1)"
            i=$((i + 1))
        done
    done

    echo
    echo "=== mem avail at end: $(mem_avail_mb) MB ==="
    echo "=== done ==="
} > "$OUT" 2>&1

echo "wrote $OUT"
