# hephaestus comparative CPU/GPU baselines

Harness: `crates/hephaestus-wgpu/benches/comparative.rs` (`cargo bench --bench comparative -p hephaestus-wgpu`).
Methodology: 50 iterations, wall-time divided by iteration count, including GPU synchronization (`poll(wgpu::PollType::Wait)`) on the host side.
Reduction rows were refreshed with `HEPHAESTUS_BENCH_DISABLE_CUDA=1` because the CUDA-enabled comparative harness terminates after the first CUDA timing on this host before reaching the reduction section.
Synchronization profile harness: `crates/hephaestus-wgpu/benches/decomposition_sync.rs` (`cargo bench --bench decomposition_sync -p hephaestus-wgpu`).
Focused sparse harness: `crates/hephaestus-wgpu/benches/sparse_comparative.rs` (`cargo bench --bench sparse_comparative -p hephaestus-wgpu`).
Inputs: Contiguous `f32` vectors/matrices of varying shapes (scaled to prevent overflow).
Machine Class: Windows 11 x86_64 dev workstation (GeForce RTX 5080).

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

| Profile | Measured floor |
| --- | --- |
| **Blocked LU 66x66 transfer/synchronization floor** | 321.4 µs |
| **Blocked QR 70x35 transfer/synchronization floor** | 213.2 µs |
| **Blocked QR 70x35 CPU panel lower bound** | 26.4 µs |
| **Blocked QR one-pass panel timestamp total** | 7.8 µs |
| **Blocked QR one-pass panel timestamp median** | 192 ns |

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
