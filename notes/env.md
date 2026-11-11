# Phase 1: environment

Branch `ai/simd-experiment`, cut from `upstream/master` = `b5cb8f6f470a3a4d6f329bbb050e9a34fde6638d` (identical to `origin/master` at start).

## Host

```
$ uname -a
Linux puneet-IdeaPad-5-15IAL7 7.0.0-43-generic #43-Ubuntu SMP PREEMPT_DYNAMIC Fri Oct 23 13:25:25 UTC 2026 x86_64 GNU/Linux
```

```
$ lscpu
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           39 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  12
On-line CPU(s) list:                     0-11
Vendor ID:                               GenuineIntel
Model name:                              12th Gen Intel(R) Core(TM) i5-1235U
CPU family:                              6
Model:                                   154
Thread(s) per core:                      2
Core(s) per socket:                      10
Socket(s):                               1
Stepping:                                4
CPU(s) scaling MHz:                      67%
CPU max MHz:                             4400.0000
CPU min MHz:                             400.0000
BogoMIPS:                                4992.00
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb intel_pt sha_ni xsaveopt xsavec xgetbv1 xsaves split_lock_detect user_shstk avx_vnni dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp hwp_pkg_req hfi vnmi umip pku ospke waitpkg gfni vaes vpclmulqdq rdpid movdiri movdir64b fsrm md_clear serialize tsxldtrk pku
Virtualization:                          VT-x
L1d cache:                               352 KiB (10 instances)
L1i cache:                               576 KiB (10 instances)
L2 cache:                                6.5 MiB (4 instances)
L3 cache:                                12 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-11
```

Core topology (from `/sys/devices/system/cpu/cpuN/topology/thread_siblings_list` and `cpufreq/cpuinfo_max_freq`):

| CPUs | kind | SMT siblings | max freq |
|---|---|---|---|
| 0-1 | P core | 0-1 | 4.40 GHz |
| 2-3 | P core | 2-3 | 4.40 GHz |
| 4-11 | E core | none | 3.30 GHz |

Benchmarks pin to `taskset -c 0,2`: one hardware thread per P core, no SMT sharing.

GPU power/clock state: `scaling_governor = powersave` on all CPUs (checked cpu0, cpu1, cpu4, cpu11). Absolute numbers drift with thermal/power state; all comparisons are A/B inside one session.

## `-march=native`

```
$ gcc -march=native -dM -E -x c /dev/null | grep -E '__AVX|__FMA|__F16C|__BMI|__AVX512|__SSE4|__POPCNT' | sort
#define __AVX2__ 1
#define __AVXVNNI__ 1
#define __AVX__ 1
#define __BMI2__ 1
#define __BMI__ 1
#define __F16C__ 1
#define __FMA__ 1
#define __POPCNT__ 1
#define __SSE4_1__ 1
#define __SSE4_2__ 1
```

So `-march=native` is `alderlake`: AVX2, FMA, F16C, AVX-VNNI, BMI1/BMI2, no `__AVX512F__`. This is the reason `ggml_cpu_has_avx512()` is false and every AVX-512 gated selector is off on this machine.

## Toolchain

No sudo (`sudo -n true` -> "sudo: interactive authentication is required").

| tool | version | path |
|---|---|---|
| cmake | 4.4.4 | `~/.local/bin/cmake` (pip `--user --break-system-packages`; plain `--user` is refused by PEP 668) |
| gcc / g++ | 15.2.0 (Ubuntu 15.2.0-16ubuntu1) | `/usr/bin/gcc`, `/usr/bin/g++` |
| clang / clang++ | 21.1.8 | `~/opt/clang-21/bin/` |
| clang-tidy | 21.1.8 | `~/opt/clang-21/bin/clang-tidy` |
| scan-build | 21.1.8 | `~/opt/clang-21/bin/scan-build` |
| python3 | 3.14.4 | `/usr/bin/python3` |
| perf | present as binary, unusable (`perf_event_paranoid = 4`) | `/usr/bin/perf` |
| cppcheck | MISSING | build from source attempted in phase 6 |
| taskset | present | `/home/linuxbrew/.linuxbrew/bin/taskset` |

Clang came from GitHub release `llvmorg-21.1.8`, asset `LLVM-21.1.8-Linux-X64.tar.xz` (2019005384 bytes). Note: the plan expected the older asset name pattern `clang+llvm-*-x86_64-linux-gnu-ubuntu-*.tar.xz`, which no longer exists in 21.1.x; the linux x86_64 toolchain is now published as `LLVM-<version>-Linux-X64.tar.xz` with top-level directory `LLVM-21.1.8-Linux-X64`. No zig fallback was needed.

`huggingface-cli` is unusable: `~/.local/bin/huggingface-cli` exists but `python3 -c "import huggingface_hub"` fails with `ModuleNotFoundError: No module named 'huggingface_hub'`. All model downloads used `curl`.

## Builds

Shared flags for every configuration: `-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF`.

| dir | compiler | build type | extra flags |
|---|---|---|---|
| `build/release` | gcc | Release | `-DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON` |
| `build/asan` | gcc | RelWithDebInfo | `-DGGML_NATIVE=ON -DLLAMA_SANITIZE_ADDRESS=ON -DLLAMA_SANITIZE_UNDEFINED=ON` |
| `build/minfeat` | gcc | Release | `-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_AVX_VNNI=OFF` |
| `build/gcc-o2` | gcc | Release | `-DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG" -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG" -DGGML_NATIVE=ON` |
| `build/clang-o2` | clang | Release | as gcc-o2, plus `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` |
| `build/clang-o3` | clang | Release | as clang-o2 with `-O3` |

Exact commands (all from the repo root):

```sh
CM=~/.local/bin/cmake
$CM -B build/release -S . -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
$CM -B build/asan -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_NATIVE=ON -DLLAMA_SANITIZE_ADDRESS=ON -DLLAMA_SANITIZE_UNDEFINED=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
$CM -B build/minfeat -S . -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_AVX_VNNI=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
$CM -B build/gcc-o2 -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG" -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG" -DGGML_NATIVE=ON ...
CC=~/opt/clang-21/bin/clang CXX=~/opt/clang-21/bin/clang++ $CM -B build/clang-o2 -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG" -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG" -DGGML_NATIVE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ...
CC=~/opt/clang-21/bin/clang CXX=~/opt/clang-21/bin/clang++ $CM -B build/clang-o3 -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" -DGGML_NATIVE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ...
```

`-DGGML_BUILD_TESTS=ON` fails: `ggml/CMakeLists.txt:314` does `add_subdirectory(tests)` and `ggml/tests` does not exist in this tree. Root-level `-DLLAMA_BUILD_TESTS=ON` (default ON for the master project) is the working switch.

## Models

| file | bytes | sha256 | source URL |
|---|---|---|---|
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 807694464 | `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` | `https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf` |
| `Llama-3.2-3B-Instruct-IQ4_XS.gguf` | 1829110304 | `b932f6fd54e64a93f8ad831439adfa2a574315fe4fd98b76c850b24029ec4998` | `https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-IQ4_XS.gguf` |
| `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | 4683074240 | `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423` | `https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf` |
| `wikitext-2-raw-v1.zip` | 4721645 | `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11` | `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip` |

All four URLs returned HTTP 200 on a HEAD check before download; no substitution was needed. Perplexity file: `models/wikitext-2-raw/wiki.test.raw` (1290590 bytes) from that zip.

## Resources at planning/benchmark time

```
MemTotal:       16088344 kB   (~15.3 GiB)
MemAvailable:    5596288 kB   (~5.3 GiB)   <- ranged 2.6-5.6 GiB during setup
SwapTotal:      15695868 kB
/dev/nvme0n1p6  390G  348G   23G  94% /home
/dev/nvme0n1p5   63G   59G  3.5G  95% /        <- /tmp lives here, keep it small
cat /proc/sys/kernel/perf_event_paranoid -> 4
```

`perf_event_paranoid = 4` and no sudo means `perf record` cannot be used; hotness evidence comes from a temporary rdtsc-instrumented build (patch kept as `notes/instrumentation.diff`, not committed) plus analytical counts.

Disk contingency: `models/` (7.3 GB) is gitignored (`.gitignore:69`), `build*` is gitignored (`.gitignore:46`). If `/home` free space drops below 5 GB, drop the least needed build directory; `/tmp` (root fs, 3.5 GB free) is only used for logs.
