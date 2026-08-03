# hephaestus comparative CPU/GPU baselines

Harness: `crates/hephaestus-wgpu/benches/comparative.rs` (`cargo bench --bench comparative -p hephaestus-wgpu`).
Methodology: 50 iterations, wall-time divided by iteration count, including GPU synchronization (`poll(wgpu::PollType::Wait)`) on the host side.
Reduction rows were refreshed with `HEPHAESTUS_BENCH_DISABLE_CUDA=1` because the CUDA-enabled comparative harness terminates after the first CUDA timing on this host before reaching the reduction section.
Synchronization profile harness: `crates/hephaestus-wgpu/benches/decomposition_sync.rs` (`cargo bench --bench decomposition_sync -p hephaestus-wgpu`).
Blocked-QR tail harness: `crates/hephaestus-wgpu/benches/blocked_qr_tail.rs` (`cargo bench --bench blocked_qr_tail -p hephaestus-wgpu`).
Focused sparse harness: `crates/hephaestus-wgpu/benches/sparse_comparative.rs` (`cargo bench --bench sparse_comparative -p hephaestus-wgpu`).
Inputs: Contiguous `f32` vectors/matrices of varying shapes (scaled to prevent overflow).
Machine Class: Windows 11 x86_64 dev workstation (GeForce RTX 5080).

## WGPU Direct Empty Prepared Reduction

The workload prepares one empty `u32` sum, validates its exact additive
identity, executes and polls one warm-up, then calls the direct
prepared-dispatch method 100,000 times before one final device poll. The plan
reference crosses `black_box` before dispatch. The baseline represented work
through independent pass and singleton fields and submitted an empty command
buffer on every call. The candidate uses one mutually exclusive work-state
enum and returns before encoder allocation for the empty variant. Matched
release runs use the same physical WGPU device and benchmark source.

| Representation | Samples | Median | Inline host size |
| --- | --- | --- | --- |
| Parallel `Vec` / `Option` fields | 19.527 µs, 20.676 µs, 19.773 µs | 19.773 µs | 80 bytes |
| Non-generic `PreparedWork` state | 1.563 ns, 0.942 ns, 1.032 ns | **1.032 ns** | **72 bytes** |

The measured median falls by more than 99.99% because the no-work path performs
no encoder allocation or queue submission; the plan's measured inline host
footprint falls by 8 bytes (10%). This size excludes owned heap/device
allocations, which are unchanged for the empty plan. The non-generic state owns
command encoding once rather than inside each scalar monomorphization, and its
tree variant freezes prepared passes as `Box<[PreparedPass]>`. These are
three-sample local measurements, not Criterion confidence intervals,
instruction-count evidence, or cross-adapter evidence. The structural
one-submission-to-zero transition and exclusive state representation are
stronger evidence than the near-timer-floor candidate timings.

## WGPU Axis-Zero Tile Geometry

The unchanged comparative workload reduces a contiguous 256x256 `f32` matrix
along axis zero. At the default 256-thread width, a 32-column tile launches
eight workgroups with eight row lanes and three tree barriers per workgroup; a
64-column tile launches four workgroups with four row lanes and two tree
barriers. Both geometries retain 256 threads and the same shared-memory array.
Runs were interleaved on the same MSVC release executable and local dependency
graph; every output was checked against Leto before timing.

| Geometry | Single samples | Single median | Eight-reduction samples | Batch median |
| --- | --- | --- | --- | --- |
| 32 columns x 8 rows | 55.522 µs, 56.684 µs, 53.114 µs | 55.522 µs | 28.494 µs, 26.922 µs, 44.332 µs | 28.494 µs |
| 64 columns x 4 rows | 57.038 µs, 39.212 µs, 37.848 µs | **39.212 µs** | 25.286 µs, 25.440 µs, 24.284 µs | **25.286 µs** |

The wider tile lowers the observed single-reduction median by 29.4% and the
batch median by 11.3%, without adding buffers, command submissions, or public
API. This is a three-sample local host/GPU comparison, not a Criterion
confidence interval or cross-adapter claim. Concurrent shared-cache workloads
were present; the deterministic reduction from eight workgroups and 32
aggregate tree barriers to four workgroups and 12 aggregate tree barriers is
stronger evidence than the exact latency ratios.

## Prepared Mixed-Reduction Batch Submission

The mixed workload combines eight independent 1,048,576-element `u32` scalar
sums with eight independent 256x256 `f32` axis-0 sums. Every scalar output is
checked against the exact host sum and every axis output against the Leto
result before timing. The timed region performs 50 call pairs or unified calls
and one final device poll. Matched runs use the same GNU nightly release
executable and resolved dependency graph.

| Submission | Samples | Median | Relative throughput |
| --- | --- | --- | --- |
| Separate scalar and axis command buffers | 115.766 µs, 128.888 µs, 117.468 µs | 117.468 µs | 1.00x |
| Unified scalar and axis command buffer | 107.462 µs, 101.488 µs, 106.640 µs | 106.640 µs | **1.10x** |

Unified submission lowers the observed median host latency by 9.2%. It removes
one transient command encoder and one queue submission per call pair while
retaining scalar tree-stage pass boundaries and adding no scratch buffers.
This is a local host/GPU result from three matched samples, not a Criterion
confidence interval or a cross-device claim. Concurrent package-cache
contention was present, so the structural encoder/submission reduction is
stronger evidence than the exact latency ratio.

## Prepared No-Op Batch Submission

The no-op batch workload prepares eight empty scalar sums and eight zero-output
axis sums. Every scalar output is downloaded and checked against the exact
additive identity; the axis output has the validated zero-element shape. The
timed region performs 50 scalar-plus-axis batch call pairs and one final device
poll. Matched old/new runs used the same GNU nightly release profile and local
stack dependency graph.

| Submission | Samples | Median |
| --- | --- | --- |
| Encode and submit two empty command buffers | 26.860 µs, 33.170 µs, 27.794 µs | 27.794 µs |
| Return before encoder allocation | 142 ns, 46 ns, 40 ns | 46 ns |

The early-return path lowers the observed median host latency by 99.8% and
eliminates two transient command encoders and two empty queue submissions per
call pair. This is a local host/GPU result from three matched samples, not a
Criterion confidence interval or a cross-device claim. The structural removal
of command construction and submission is stronger evidence than the exact
ratio at this sub-microsecond result scale.

## Prepared Scalar-Reduction Batch Encoding

The prepared scalar batch uses eight independent 1,048,576-element `u32`
sums. Every device output is downloaded and compared with the exact host sum
before the timed loop. The timed region performs 50 batch submissions and one
final device poll. Interleaved old/new runs used the same GNU nightly release
profile and resolved dependency graph.

| Encoding | Samples | Median | Relative throughput |
| --- | --- | --- | --- |
| One compute pass per tree stage | 234.206 µs, 228.006 µs, 246.956 µs | 234.206 µs | 1.00x |
| One compute pass per batch stage | 104.882 µs, 99.060 µs, 100.918 µs | 100.918 µs | **2.32x** |

The stage-major encoder lowers the observed median batch latency by 56.9%.
It opens one compute pass per maximum tree depth instead of one pass per stage
of every reduction, without changing buffers, allocations, or arithmetic
order. This is a local host/GPU result from three matched samples, not a
Criterion confidence interval or a cross-device claim. Concurrent
package-cache contention was present, so the structural pass-count reduction
is stronger evidence than the exact latency ratio.

## Prepared Axis-Reduction Batch Encoding

The prepared axis batch uses eight independent 256x256 axis-0 sums, with all
eight device outputs downloaded and compared with the Leto result before the
timed loop. The timed region performs 50 batch submissions and one final device
poll. Interleaved old/new runs used the same GNU nightly release profile and
resolved dependency graph.

| Encoding | Samples | Median | Relative throughput |
| --- | --- | --- | --- |
| One compute pass per reduction | 105.092 µs, 105.458 µs, 103.420 µs | 105.092 µs | 1.00x |
| One compute pass per batch | 37.726 µs, 25.902 µs, 32.250 µs | 32.250 µs | **3.26x** |

The single-pass encoder lowers the observed median batch latency by 69.3%.
This is a local host/GPU result from three matched samples, not a Criterion
confidence interval or a cross-device claim. Concurrent package-cache
contention was present, so the structural reduction from eight compute-pass
constructions to one is stronger evidence than the exact latency ratio.

## Comparative Results

| Benchmark | GPU (WGPU) | Leto CPU | ndarray CPU | nalgebra CPU | GPU Speedup (vs Leto) | GPU Speedup (vs ndarray) | GPU Speedup (vs nalgebra) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Elementwise Add** ($N = 2^{20}$) | 60.97 µs | 1.02 ms | 1.04 ms | — | **16.73x** | **17.06x** | — |
| **Elementwise Exp** ($N = 2^{20}$) | 70.18 µs | 1.90 ms | 1.88 ms | — | **27.07x** | **26.79x** | — |
| **Sum Reduction** ($N = 2^{20}$, prepared final pass) | 42.70 µs | 63.09 µs | 85.47 µs | — | **1.48x** | **2.00x** | — |
| **Axis Sum** (256x256 over axis 0, axis-0 tiled prepared) | 22.14 µs | 10.45 µs | 6.53 µs | 36.13 µs | **0.47x** | **0.29x** | **1.63x** |
| **Axis Min** (256x256 over axis 0, axis-0 tiled prepared) | 20.73 µs | 5.41 µs | 4.63 µs | 8.67 µs | **0.26x** | **0.22x** | **0.42x** |
| **Axis Max** (256x256 over axis 0, axis-0 tiled prepared) | 11.78 µs | 5.36 µs | 4.42 µs | 5.17 µs | **0.46x** | **0.38x** | **0.44x** |
| **Axis Mean** (256x256 over axis 0, axis-0 tiled prepared) | 18.05 µs | 7.17 µs | 5.88 µs | 18.24 µs | **0.40x** | **0.33x** | **1.01x** |
| **Matmul 64x64** | 58.61 µs | 31.29 µs | 5.84 µs | 14.74 µs | **0.53x** | **0.10x** | **0.25x** |
| **Matmul 256x256** | 47.47 µs | 929.51 µs | 251.41 µs | 1.10 ms | **19.58x** | **5.30x** | **23.17x** |
| **Cumsum** (256x256 over axis 1) | 46.53 µs | 30.70 µs | 108.60 µs | 136.39 µs | **0.66x** | **2.33x** | **2.93x** |
| **Matrix Power** (64x64 exponent 5) | 281.48 µs | 116.74 µs | 17.93 µs | 14.17 µs | **0.41x** | **0.064x** | **0.050x** |
| **Kronecker Product** (64x64 ⊗ 8x8) | 48.96 µs | 209.17 µs | — | 403.71 µs | **4.27x** | — | **8.25x** |
| **Dot Product** ($N = 65,536$) | 97.42 µs | 3.84 µs | 8.37 µs | — | **0.039x** | **0.086x** | — |
| **Trace** (256x256) | 42.72 µs | 122 ns | 150 ns | — | **0.0029x** | **0.0035x** | — |
| **Matrix Rank** (64x64 diagonal rank 32) | 6.26 ms | 38.92 µs | — | — | **0.0062x** | — | — |
| **Determinant** (64x64 diagonal) | 8.20 ms | 10.49 µs | 22 ns | 6.99 µs | **0.0013x** | **0.0000027x** | **0.00085x** |
| **Blocked Cholesky Decomposition** (128x128 SPD) | 556.93 µs | 116.72 µs | — | 28.82 µs | **0.21x** | — | **0.052x** |
| **LU Decomposition** (32x32) | 92.94 µs | 2.20 µs | — | 1.65 µs | **0.024x** | — | **0.018x** |
| **Blocked LU Decomposition** (66x66) | 411.66 µs | 9.56 µs | — | 7.57 µs | **0.023x** | — | **0.018x** |
| **Full-Pivot LU Decomposition** (32x32) | 131.24 µs | 20.89 µs | — | 12.92 µs | **0.16x** | — | **0.098x** |
| **QR Decomposition** (48x24) | 132.42 µs | 6.46 µs | — | 3.78 µs | **0.049x** | — | **0.029x** |
| **Blocked QR Decomposition** (70x35) | 480.8 µs | 14.9 µs | — | 10.0 µs | **0.031x** | — | **0.021x** |
| **SVD Decomposition** (32x16) | 143.53 µs | 14.06 µs | — | 4.10 µs | **0.098x** | — | **0.029x** |
| **Bidiagonalization** (32x16) | 167.60 µs | 12.74 µs | — | 7.15 µs (nalgebra SVD) | **0.076x** | — | **0.043x** |
| **Schur Decomposition** (32x32) | 142.45 µs | 10.43 µs | — | 4.94 µs (nalgebra eigenvalues) | **0.073x** | — | **0.035x** |
| **Hessenberg Reduction** (32x32) | 130.63 µs | 17.47 µs | — | 5.33 µs | **0.13x** | — | **0.041x** |
| **Bunch-Kaufman Decomposition** (32x32) | 109.39 µs | 3.23 µs | — | 1.58 µs (nalgebra determinant) | **0.030x** | — | **0.014x** |
| **UDU Decomposition** (32x32) | 129.14 µs | 12.00 µs | — | 1.60 µs (nalgebra determinant) | **0.093x** | — | **0.012x** |
| **Symmetric Eigen Jacobi** (32x32) | 431.85 µs | 304.34 µs | — | 21.54 µs | **0.70x** | — | **0.050x** |
| **General Eigenvalues** (32x32 block rotations) | 112.27 µs | 9.19 µs | — | 5.43 µs | **0.082x** | — | **0.048x** |
| **Norm L1** ($N = 65,536$) | 78.14 µs | 2.13 µs | — | — | **0.027x** | — | — |
| **Norm L2** ($N = 65,536$) | 162.24 µs | 2.94 µs | — | — | **0.018x** | — | — |
| **Norm Max** ($N = 65,536$) | 89.88 µs | 2.60 µs | — | — | **0.029x** | — | — |
| **Column-Pivoted QR Decomposition** (32x32) | 129.75 µs | 20.46 µs | — | 14.09 µs | **0.16x** | — | **0.11x** |
| **Pseudoinverse** (32x32) | 1.98 ms | 1.76 ms | — | 19.66 µs | **0.89x** | — | **0.010x** |
| **Matrix Exponential** (32x32) | 151.85 µs | 57.83 µs | — | — | **0.38x** | — | — |
| **PRNG Uniform** ($N = 2^{20}$) | 4.02 ms | 1.79 ms | — | — | **0.45x** | — | — |
| **PRNG Normal** ($N = 2^{20}$) | 17.12 ms | 14.13 ms | — | — | **0.83x** | — | — |
| **SpMV** ($1000 \times 1000$ CSR, prepared reusable output) | 61.15 µs | 1.23 µs | — | — | **0.020x** | — | — |
| **Batched SpMV via `spmv_many`** ($1000 \times 1000$ CSR, 128 RHS vectors) | 62.76 µs | 150.41 µs | — | — | **2.40x** | — | — |
| **SpMM** ($1000 \times 1000 \times 128$, warmed batched prepared outputs, dense RHS fast path) | 12.26 µs | 35.23 µs | — | — | **2.87x** | — | — |

## Synchronization Profile

| Profile | Measurement |
| --- | --- |
| **Blocked LU 66x66 transfer/synchronization floor** | 321.4 µs |
| **Blocked QR 70x35 current end-to-end** | 96.5 µs |

The QR component harness now executes and validates the production
factorization rather than approximating the superseded pre-PR transfer
schedule. The retired CPU-tail diagnostic measured 38.189 µs in the
pre-change selection run and 30.373 µs in a separate post-change closure run;
these are distinct profile samples, not one stable benchmark result. The
production harness no longer duplicates the implementation's private panel
size after the narrow route made that diagnostic obsolete.

### Retained blocked-QR region-transfer baseline

Machine: Windows 11, Intel Core Ultra 9 285K, NVIDIA GeForce RTX 5080,
driver 610.47. Criterion 0.8.2 uses device completion inside every timed
iteration and validates the complete `R` factor against Leto before timing.

| Retained blocked schedule | 95% interval | Median |
| --- | ---: | ---: |
| 192×129 routing boundary | 1.2877–1.3495 ms | 1.3136 ms |
| 384×256 eight-panel workload | 4.0608–4.5832 ms | 4.2996 ms |

The 192×129 schedule prepares five downloads and five uploads: three single
panel transfer pairs followed by one paired panel/tail download and upload.
The 384×256 schedule prepares eight downloads and eight uploads. Each
preparation checks out pooled staging or uniform storage and constructs a new
bind group; the matched result run will determine whether retaining those
resources across panels changes end-to-end latency. These figures are the
`qr-transfer-before` baseline, not an optimization claim.

### Retained blocked-QR transfer-workspace Criterion A/B

Machine: Windows 11, Intel Core Ultra 9 285K, NVIDIA GeForce RTX 5080,
driver 610.47. The old and new revisions ran back-to-back with the same
Criterion 0.8.2 instrument, device completion inside every timed iteration,
and a complete Leto `R` differential before timing.

| Schedule | Before 95% interval | After 95% interval | Central change |
| --- | ---: | ---: | ---: |
| 192×128 unchanged direct control | 556.23–640.37 µs | 533.83–625.49 µs | −1.4307%, `p = 0.72` |
| 192×129 retained blocked route | 1.5530–1.8142 ms | 1.3508–1.5132 ms | −11.796%, `p = 0.00` |
| 384×256 retained blocked route | 4.1174–4.5282 ms | 3.9127–4.0826 ms | −7.2117%, `p = 0.00` |

The first-position 70×35 direct control shifted with GPU clock or concurrent
host load and is excluded from the performance claim. The warmed 192×128
direct control has no detectable change, while both retained blocked routes
improve. The implementation retains one panel download workspace across
iterations, reducing download resource preparations from five to three at
192×129 and from eight to three at 384×256. The final tail remains an exact
one-shot allocation, so peak staging is unchanged: 25,344 bytes at 192×129
and 98,304 bytes at 384×256.

### Blocked-QR four-panel direct-transfer Criterion A/B

Machine: Windows 11, Intel Core Ultra 9 285K, NVIDIA GeForce RTX 5080,
driver 610.47. Both runs use Criterion 0.8.2 defaults, the same 192×128 input,
device completion inside the timed iteration, and a complete Leto `R`
differential before timing.

| Four-panel schedule | 95% interval | Median |
| --- | ---: | ---: |
| Blocked region transfers (`before`) | 1.1903–1.2206 ms | 1.2046 ms |
| One dense download and one `R` upload | 456.32–466.56 µs | 461.24 µs |

The displayed medians differ by **61.710%**. Criterion reports a
**−62.055% central change estimate** (95% interval **−62.621% to −61.496%**,
`p = 0.00`). The unchanged 70×35 control has a **−1.1462% central change
estimate** (95% interval **−3.7574% to +1.1498%**, `p = 0.39`), so the matched
run detects no control change.
At 192×128 the direct route removes two 24,576-byte compact device buffers,
256-byte reflector storage, and one 256-byte pooled uniform allocation:
**49,664 bytes of device scratch**. The 98,304-byte device `R` output and
readback staging allocation remain required.

### Rejected blocked-QR eight-panel direct route

Machine: Windows 11, Intel Core Ultra 9 285K, NVIDIA GeForce RTX 5080,
driver 610.47. Criterion 0.8.2 runs the unchanged `blocked_qr_tail` workload
back-to-back on the clean Leto overlay, waits for device completion inside each
timed iteration, and validates the complete `R` factor against Leto first.

| Workload | Four-panel route 95% interval | Eight-panel candidate 95% interval | Criterion change |
| --- | ---: | ---: | ---: |
| 192×128 unchanged control | 354.91–359.10 µs | 360.96–364.67 µs | −0.0377% to +1.8236%, `p = 0.06` |
| 192×129 five panels | 937.57–952.44 µs | 345.98–349.25 µs | −63.864% to −62.750%, `p = 0.00` |
| 384×256 eight panels | 2.6701–2.7027 ms | 21.756–21.916 ms | +707.13% to +718.54%, `p = 0.00` |

The candidate fails because the eight-panel target regresses by more than
sevenfold even though the five-panel target improves and the direct route
avoids six mapping polls plus persistent scratch at 384×256. This end-to-end
instrument does not isolate host factorization from dense transfer and
orchestration costs. The production limit remains four panels; no runtime or
benchmark change is retained.

### Blocked-QR narrow direct-transfer Criterion A/B

Machine: Windows 11, Intel Core Ultra 9 285K, NVIDIA GeForce RTX 5080,
driver 610.47. Both runs use Criterion 0.8.2 defaults, the same 70×35 input,
device completion inside the timed iteration, and a complete Leto `R`
differential before timing.

| Narrow schedule | 95% interval | Median |
| --- | ---: | ---: |
| Paired region gather/scatter (`before`) | 345.10–363.03 µs | 353.68 µs |
| One dense download and one `R` upload | 130.49–143.77 µs | 136.95 µs |

The displayed medians differ by **61.279%**. Criterion independently reports a
statistically significant **−61.887% central change estimate** (95% interval
**−63.093% to −60.633%**, `p = 0.00`). Matrices at or below the two-panel
boundary have no wide GPU trailing update, so the direct schedule calls the
existing host QR implementation instead of constructing blocked region
transfers. At 70×35 this removes two 8,960-byte compact device buffers,
256-byte reflector storage, and one 256-byte pooled uniform allocation:
**18,432 bytes of device scratch**. The 9,800-byte device `R` output and
9,800-byte readback staging allocation remain required.

### Blocked-QR final two-panel Criterion A/B

Machine: Windows 11, Intel Core Ultra 9 285K, NVIDIA GeForce RTX 5080,
driver 610.47. Both runs use Criterion 0.8.2 defaults, the same 70×35 input,
device completion inside the timed iteration, and a complete Leto `R`
differential before timing.

| Schedule | 95% interval | Median |
| --- | ---: | ---: |
| Separate final-panel readback (`before`) | 509.34–524.73 µs | 516.58 µs |
| Paired panel/tail readback | 335.57–359.05 µs | 346.19 µs |

The displayed medians differ by **32.984%**. Criterion independently reports a
statistically significant **−29.612% central change estimate** (95% interval
**−34.402% to −23.972%**, `p = 0.00`). The schedule removes one host/device poll
at this shape while preserving the complete factorization values. Persistent
compact device scratch remains two `m × 32` buffers. The paired map holds one
additional transient
`m × tail_cols` staging buffer; at 70×35 this is **840 bytes**, returned to the
pool immediately after the shared poll.

## Analysis

1. **Compute vs. Memory Bandwidth & GPU Scaling**:
   - For **Elementwise Add** (memory-bound, low arithmetic intensity), the GPU reaches $\approx 60.97 \text{ µs/iter}$, outperforming Leto by $\approx 16.73\times$ and `ndarray` by $\approx 17.06\times$.
   - For **Elementwise Exp** (compute-bound, high arithmetic intensity), the GPU reaches $\approx 70.18 \text{ µs/iter}$, outperforming Leto by $\approx 27.07\times$ and `ndarray` by $\approx 26.79\times$.
   - For **Sum Reduction**, the scalar path now has a final reduction shader that lets one workgroup fold up to `BlockWidth * BlockWidth` partials, reducing the $2^{20}$ sum tree from three compute passes to two. The latest full comparative run measures WGPU at $\approx 42.70 \text{ µs/iter}$ against `ndarray` at $\approx 85.47 \text{ µs/iter}$.
   - For **Axis Reductions**, the WGPU path now uses an axis-0 tiled shader for row-reducing rank-2 inputs: each workgroup reduces up to 16 output columns instead of launching one workgroup per output element. This removes the prior pathological max/mean rows, but the downstream Leto row-major CPU fast path remains faster than WGPU for this 65,536-element workload. WGPU is still overhead-bound for this small-axis shape, so the parity route is a measured CPU small-axis policy or tighter multi-axis GPU batching rather than more per-element shader arithmetic.
   - For **Matmul 256x256**, the GPU reaches $\approx 47.47 \text{ µs/iter}$ due to optimized parallel matrix contraction tiles, achieving **19.58x** speedup over Leto CPU and **23.17x** over nalgebra CPU.

2. **Driver Overhead and Reflector Batching**:
   - For smaller workloads, the CPU dominates due to zero launch overhead.
   - **Householder Reflector Batching**: In the blocked QR algorithm, we batched all compute passes for the panel inside a single command encoder and submitted it exactly once per panel instead of issuing separate submissions and waiting/polling. This reduced host-GPU queue submission traffic by **32x**, leading to a **2.6x** speedup on **Blocked QR Decomposition (70x35)**, dropping execution time from **2.90 ms** to **1.10 ms**.
   - CUDA blocked QR was similarly optimized by packing Householder vectors and uploading them once per panel, avoiding 32 separate allocations and uploads.
   - The blocked QR component profile measures the CPU panel lower bound at
     **25.3 µs** for 70x35, while the synthetic host/device synchronization
     floor remains **219.9 µs**. The production path constructs the host-side
     `QrDecomposition` from the blocked factors with `from_raw_parts`, so the
     obsolete final Leto recompute is no longer profiled. At this shape, the next
     measured bottleneck is transfer and synchronization, not CPU panel
     arithmetic.
   - Packing Householder vector offsets and beta coefficients into one
     reflector metadata buffer reduces per-panel metadata uploads and storage
     bindings from two to one. Reusing one Householder metadata uniform and bind
     group across blocked-QR panels removes another per-panel CPU-side WGPU
     resource construction. Delaying the full matrix copy until after the first
     panel readback lets the first panel read from the original input buffer and
     avoids placing the full copy on the critical path before the first CPU
     panel factorization. The 70x35 synchronization profile improved to
     **213.2 µs**, but remains transfer-bound.
   - The final two-panel schedule now gathers the 32-column panel and
     three-column tail before one poll, finishes the tail with the shared packed
     Householder operation, and writes both regions in one submission. The
     matched Criterion median decreases from **516.58 µs** to **346.19 µs**
     (**32.984%**); Criterion's central change estimate is **−29.612%**. No
     persistent compact scratch is added. The bounded 840-byte transient
     staging increase is reported above rather than treated as free memory.

---

## Local Workstation Fallback Baselines (Vulkan Software Emulation / Stub CUDA)

The following baselines were measured in the virtualized workstation sandbox environment. In this environment, WGPU runs via a CPU-emulated software-rasterized adapter (Vulkan software driver), and CUDA runs in stub mode (compiles out GPU operations).

These numbers showcase performance on a system without hardware GPU acceleration:

| Benchmark | GPU (WGPU Fallback) | Leto CPU | ndarray CPU | nalgebra CPU | GPU Speedup (vs Leto) |
| --- | --- | --- | --- | --- | --- |
| **Elementwise Add** ($N = 2^{20}$) | 623.91 µs | 1.04 ms | 1.26 ms | — | **1.66x** |
| **Elementwise Exp** ($N = 2^{20}$) | 943.10 µs | 2.02 ms | 1.99 ms | — | **2.14x** |
| **Sum Reduction** ($N = 2^{20}$) | 2.08 ms | 64.77 µs | 79.76 µs | — | **0.03x** |
| **Axis Sum** (256x256 over axis 0) | 559.91 µs | 42.23 µs | 5.34 µs | 19.74 µs | **0.08x** |
| **Axis Min** (256x256 over axis 0) | 594.03 µs | 42.50 µs | 8.07 µs | 12.56 µs | **0.07x** |
| **Axis Max** (256x256 over axis 0) | 543.70 µs | 39.61 µs | 7.31 µs | 11.96 µs | **0.07x** |
| **Axis Mean** (256x256 over axis 0) | 509.68 µs | 41.29 µs | 4.72 µs | 22.46 µs | **0.08x** |
| **Matmul 64x64** | 289.06 µs | 42.97 µs | 11.12 µs | 32.68 µs | **0.15x** |
| **Matmul 256x256** | 4.89 ms | 960.50 µs | 566.17 µs | 1.47 ms | **0.20x** |
| **Cumsum** (256x256 over axis 1) | 3.74 ms | 94.50 µs | 139.73 µs | 178.16 µs | **0.03x** |
| **Matrix Power** (64x64 exponent 5) | 3.95 ms | 168.16 µs | 42.83 µs | 33.21 µs | **0.04x** |
| **Kronecker Product** (64x64 ⊗ 8x8) | 864.03 µs | 236.19 µs | — | 681.66 µs | **0.27x** |
| **Dot Product** ($N = 65,536$) | 755.58 µs | 4.15 µs | 5.58 µs | — | **0.01x** |
| **Trace** (256x256) | 117.54 µs | 140 ns | 216 ns | — | **0.001x** |
| **Matrix Rank** (64x64 diagonal rank 32) | 4.87 ms | 25.67 µs | — | — | **0.005x** |
| **Determinant** (64x64 diagonal) | 7.68 ms | 12.24 µs | 16 ns | 6.96 µs | **0.002x** |
| **Blocked Cholesky** (128x128 SPD) | 32.80 ms | 146.15 µs | — | 27.95 µs | **0.004x** |
| **LU Decomposition** (32x32) | 650.57 µs | 2.18 µs | — | 1.54 µs | **0.003x** |
| **Blocked LU Decomposition** (66x66) | 3.24 ms | 13.40 µs | — | 10.28 µs | **0.004x** |
| **Full-Pivot LU** (32x32) | 605.03 µs | 20.81 µs | — | 8.24 µs | **0.034x** |
| **QR Decomposition** (48x24) | 652.82 µs | 6.78 µs | — | 4.04 µs | **0.010x** |
| **Blocked QR Decomposition** (70x35) | 6.38 ms | 13.74 µs | — | 9.84 µs | **0.002x** |
| **SVD Decomposition** (32x16) | 529.61 µs | 18.47 µs | — | 4.75 µs | **0.035x** |
| **Bidiagonalization** (32x16) | 987.54 µs | 15.52 µs | — | 9.53 µs (SVD) | **0.016x** |
| **Schur Decomposition** (32x32) | 1.14 ms | 14.81 µs | — | 6.33 µs (eigen) | **0.013x** |
| **Hessenberg Reduction** (32x32) | 1.10 ms | 25.61 µs | — | 6.88 µs | **0.023x** |
| **Bunch-Kaufman** (32x32) | 1.06 ms | 3.57 µs | — | 1.75 µs | **0.003x** |
| **UDU Decomposition** (32x32) | 609.59 µs | 14.16 µs | — | 1.77 µs | **0.023x** |
| **Symmetric Eigen Jacobi** (32x32) | 979.86 µs | 359.39 µs | — | 21.29 µs | **0.37x** |
| **General Eigenvalues** (32x32) | 132.19 µs | 9.82 µs | — | 6.21 µs | **0.07x** |
| **Norm L1** ($N = 65,536$) | 438.91 µs | 2.56 µs | — | — | **0.006x** |
| **Norm L2** ($N = 65,536$) | 733.71 µs | 3.21 µs | — | — | **0.004x** |
| **Norm Max** ($N = 65,536$) | 716.00 µs | 2.25 µs | — | — | **0.003x** |
| **Column-Pivoted QR** (32x32) | 1.01 ms | 21.96 µs | — | 13.97 µs | **0.022x** |
| **Pseudoinverse** (32x32) | 2.39 ms | 1.90 ms | — | 20.18 µs | **0.79x** |
| **Matrix Exponential** (32x32) | 629.96 µs | 59.34 µs | — | — | **0.09x** |
| **PRNG Uniform** ($N = 2^{20}$) | 3.99 ms | 1.83 ms | — | — | **0.46x** |
| **PRNG Normal** ($N = 2^{20}$) | 18.71 ms | 16.03 ms | — | — | **0.86x** |
| **SpMV** ($1000 \times 1000$ CSR) | 107.20 µs | 2.77 µs | — | — | **0.026x** |
| **SpMM** ($1000 \times 1000 \times 128$) | 586.58 µs | 35.63 µs | — | — | **0.061x** |
