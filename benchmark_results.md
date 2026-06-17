# hephaestus comparative CPU/GPU baselines

Harness: `crates/hephaestus-wgpu/benches/comparative.rs` (`cargo bench --bench comparative -p hephaestus-wgpu`).
Methodology: 50 iterations, wall-time divided by iteration count, including GPU synchronization (`poll(wgpu::PollType::Wait)`) on the host side.
Synchronization profile harness: `crates/hephaestus-wgpu/benches/decomposition_sync.rs` (`cargo bench --bench decomposition_sync -p hephaestus-wgpu`).
Focused sparse harness: `crates/hephaestus-wgpu/benches/sparse_comparative.rs` (`cargo bench --bench sparse_comparative -p hephaestus-wgpu`).
Inputs: Contiguous `f32` vectors/matrices of varying shapes (scaled to prevent overflow).
Machine Class: Windows 11 x86_64 dev workstation (GeForce RTX 5080).

## Comparative Results

| Benchmark | GPU (WGPU) | Leto CPU | ndarray CPU | nalgebra CPU | GPU Speedup (vs Leto) | GPU Speedup (vs ndarray) | GPU Speedup (vs nalgebra) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Elementwise Add** ($N = 2^{20}$) | 60.97 µs | 1.02 ms | 1.04 ms | — | **16.73x** | **17.06x** | — |
| **Elementwise Exp** ($N = 2^{20}$) | 70.18 µs | 1.90 ms | 1.88 ms | — | **27.07x** | **26.79x** | — |
| **Sum Reduction** ($N = 2^{20}$) | 116.97 µs | 73.48 µs | 84.22 µs | — | **0.63x** | **0.72x** | — |
| **Axis Sum** (256x256 over axis 0) | 110.33 µs | 44.84 µs | 3.34 µs | 15.80 µs | **0.41x** | **0.030x** | **0.14x** |
| **Axis Min** (256x256 over axis 0) | 94.83 µs | 49.85 µs | 4.43 µs | 5.90 µs | **0.53x** | **0.047x** | **0.062x** |
| **Axis Max** (256x256 over axis 0) | 75.13 µs | 49.06 µs | 4.28 µs | 5.49 µs | **0.65x** | **0.057x** | **0.078x** |
| **Axis Mean** (256x256 over axis 0) | 73.25 µs | 44.45 µs | 4.57 µs | 15.59 µs | **0.61x** | **0.062x** | **0.21x** |
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
| **SpMV** ($1000 \times 1000$ CSR) | 100.89 µs | 1.28 µs | — | — | **0.013x** | — | — |
| **SpMM** ($1000 \times 1000 \times 128$) | 73.63 µs | 32.47 µs | — | — | **0.44x** | — | — |

## Synchronization Profile

| Profile | Measured floor |
| --- | --- |
| **Blocked LU 66x66 transfer/synchronization floor** | 333.6 µs |
| **Blocked QR 70x35 transfer/synchronization floor** | 219.9 µs |
| **Blocked QR 70x35 CPU panel lower bound** | 25.3 µs |
| **Blocked QR 70x35 final Leto recompute** | 12.9 µs |
| **Blocked QR one-pass panel timestamp total** | 8.3 µs |
| **Blocked QR one-pass panel timestamp median** | 192 ns |

## Analysis

1. **Compute vs. Memory Bandwidth & GPU Scaling**:
   - For **Elementwise Add** (memory-bound, low arithmetic intensity), the GPU reaches $\approx 60.97 \text{ µs/iter}$, outperforming Leto by $\approx 16.73\times$ and `ndarray` by $\approx 17.06\times$.
   - For **Elementwise Exp** (compute-bound, high arithmetic intensity), the GPU reaches $\approx 70.18 \text{ µs/iter}$, outperforming Leto by $\approx 27.07\times$ and `ndarray` by $\approx 26.79\times$.
   - For **Matmul 256x256**, the GPU reaches $\approx 47.47 \text{ µs/iter}$ due to optimized parallel matrix contraction tiles, achieving **19.58x** speedup over Leto CPU and **23.17x** over nalgebra CPU.

2. **Driver Overhead and Reflector Batching**:
   - For smaller workloads, the CPU dominates due to zero launch overhead.
   - **Householder Reflector Batching**: In the blocked QR algorithm, we batched all compute passes for the panel inside a single command encoder and submitted it exactly once per panel instead of issuing separate submissions and waiting/polling. This reduced host-GPU queue submission traffic by **32x**, leading to a **2.6x** speedup on **Blocked QR Decomposition (70x35)**, dropping execution time from **2.90 ms** to **1.10 ms**.
   - CUDA blocked QR was similarly optimized by packing Householder vectors and uploading them once per panel, avoiding 32 separate allocations and uploads.
   - The blocked QR component profile measures the CPU panel lower bound at
     **25.3 µs** and the final Leto recompute at **12.9 µs** for 70x35,
     while the synthetic host/device synchronization floor remains
     **219.9 µs**. At this shape, the next measured bottleneck is transfer
     and synchronization, not CPU panel arithmetic.
   - Packing Householder vector offsets and beta coefficients into one
     reflector metadata buffer reduces per-panel metadata uploads and storage
     bindings from two to one. The 70x35 blocked QR row still trails Leto and
     `nalgebra` after this change, so this is static transfer-surface
     reduction, not measured performance parity.
