#!/bin/sh
# Interleaved before/after A/B measurement.
#
# The plain "run the same script before and after" comparison on this host carries
# a few percent of drift between runs (power/thermal state, page cache), which is
# the same order as the effect being measured. This driver alternates the two
# builds within a single session, one llama-bench rep at a time, so both sides see
# the same machine state, and reports a paired per-repetition ratio next to the
# medians.
#
# Usage: notes/run_ab_interleaved.sh BEFORE_BIN_DIR AFTER_BIN_DIR OUT.csv [REPS]
# Environment: MODEL_FILTER (substring) to run a subset.

set -eu

BEFORE=${1:?usage: run_ab_interleaved.sh BEFORE_BIN AFTER_BIN OUT.csv [REPS]}
AFTER=${2:?usage: run_ab_interleaved.sh BEFORE_BIN AFTER_BIN OUT.csv [REPS]}
OUT=${3:?usage: run_ab_interleaved.sh BEFORE_BIN AFTER_BIN OUT.csv [REPS]}
REPS=${4:-5}

PIN="taskset -c 0,2"
T=2

mkdir -p build/logs

mem_avail_mb() {
    awk '/MemAvailable/ { printf "%d", $2/1024 }' /proc/meminfo
}

# name:model:pp:tg:min_mem_mb
SPECS="1B:models/Llama-3.2-1B-Instruct-Q4_K_M.gguf:512:128:2048
3B:models/Llama-3.2-3B-Instruct-IQ4_XS.gguf:512:128:3000
7B:models/Qwen2.5-7B-Instruct-Q4_K_M.gguf:256:64:5200"

run_one() { # bindir model pp tg repack
    $PIN "$1/llama-bench" -m "$2" -p "$3" -n "$4" -t "$T" -r 1 --repack "$5" -o md 2>/dev/null \
        | awk -F'|' '{ t=$7; v=$8; gsub(/^[ \t]+|[ \t]+$/,"",t); gsub(/^[ \t]+/,"",v); split(v,a," ");
                       if (t ~ /^(pp|tg)[0-9]+$/) print t "," a[1] }'
}

echo "model,repack,rep,side,test,tps" > "$OUT"

for spec in $SPECS; do
    name=$(echo "$spec"  | cut -d: -f1)
    model=$(echo "$spec" | cut -d: -f2)
    pp=$(echo "$spec"    | cut -d: -f3)
    tg=$(echo "$spec"    | cut -d: -f4)
    need=$(echo "$spec"  | cut -d: -f5)

    if [ -n "${MODEL_FILTER:-}" ] && [ "$name" != "$MODEL_FILTER" ]; then
        continue
    fi

    avail=$(mem_avail_mb)
    if [ "$avail" -lt "$need" ]; then
        echo "SKIP $name: MemAvailable ${avail} MB < ${need} MB" >&2
        continue
    fi

    for rp in 1 0; do
        i=1
        while [ "$i" -le "$REPS" ]; do
            for side in before after; do
                if [ "$side" = "before" ]; then bin=$BEFORE; else bin=$AFTER; fi
                run_one "$bin" "$model" "$pp" "$tg" "$rp" | sed "s/^/$name,$rp,$i,$side,/" >> "$OUT"
            done
            i=$((i + 1))
        done
    done
done

echo "wrote $OUT"

awk -F, '
function isort(arr, n,   i, j, t) { for (i = 1; i <= n; i++) for (j = i+1; j <= n; j++) if (arr[j] < arr[i]) { t = arr[i]; arr[i] = arr[j]; arr[j] = t } }
function imed(arr, n) { return (n % 2) ? arr[(n+1)/2] : (arr[n/2] + arr[n/2+1]) / 2 }
NR > 1 {
    key = $1 " rpk=" $2 " " $5;
    if ($4 == "before") b[key SUBSEP $3] = $6; else a[key SUBSEP $3] = $6;
    seen[key] = 1;
}
END {
    printf "%-24s %10s %10s %8s  %s\n", "model/test", "before", "after", "ratio", "per-rep after/before";
    for (k in seen) {
        nb = 0; na = 0;
        for (r = 1; r <= 12; r++) {
            if ((k SUBSEP r) in b) bv[++nb] = b[k SUBSEP r] + 0;
            if ((k SUBSEP r) in a) av[++na] = a[k SUBSEP r] + 0;
        }
        isort(bv, nb); isort(av, na);
        mb = imed(bv, nb); ma = imed(av, na);
        s = "";
        for (r = 1; r <= 12; r++) {
            if (((k SUBSEP r) in b) && ((k SUBSEP r) in a)) {
                x = b[k SUBSEP r] + 0; y = a[k SUBSEP r] + 0;
                s = s sprintf("%.3f ", (x > 0) ? y/x : 0);
            }
        }
        printf "%-24s %10.2f %10.2f %8.4f  %s\n", k, mb, ma, (mb > 0) ? ma/mb : 0, s;
    }
}' "$OUT" | { IFS= read -r hdr; echo "$hdr"; sort; }
