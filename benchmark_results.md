# hephaestus comparative CPU/GPU baselines

Harness: `crates/hephaestus-wgpu/benches/comparative.rs` (`cargo bench --bench comparative -p hephaestus-wgpu`).
Methodology: 50 iterations, wall-time divided by iteration count, including GPU synchronization (`poll(wgpu::PollType::Wait)`) on the host side.
Synchronization profile harness: `crates/hephaestus-wgpu/benches/decomposition_sync.rs` (`cargo bench --bench decomposition_sync -p hephaestus-wgpu`).
Inputs: Contiguous `f32` vectors/matrices of varying shapes (scaled to prevent overflow).
Machine Class: Windows 11 x86_64 dev workstation (GeForce RTX 5080).

## Comparative Results

| Benchmark | GPU (WGPU) | Leto CPU | ndarray CPU | nalgebra CPU | GPU Speedup (vs Leto) | GPU Speedup (vs ndarray) | GPU Speedup (vs nalgebra) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Elementwise Add** ($N = 2^{20}$) | 79.9 µs | 1.03 ms | 1.05 ms | — | **12.86x** | **13.14x** | — |
| **Elementwise Exp** ($N = 2^{20}$) | 45.8 µs | 1.86 ms | 1.87 ms | — | **40.55x** | **40.74x** | — |
| **Sum Reduction** ($N = 2^{20}$) | 101.0 µs | 62.9 µs | 77.9 µs | — | **0.62x** | **0.77x** | — |
| **Axis Sum** (256x256 over axis 0) | 100.5 µs | 41.5 µs | 4.2 µs | 15.3 µs | **0.41x** | **0.042x** | **0.15x** |
| **Axis Min** (256x256 over axis 0) | 73.5 µs | 47.7 µs | 4.3 µs | 5.3 µs | **0.65x** | **0.059x** | **0.072x** |
| **Axis Max** (256x256 over axis 0) | 82.7 µs | 49.5 µs | 4.2 µs | 5.9 µs | **0.60x** | **0.051x** | **0.071x** |
| **Axis Mean** (256x256 over axis 0) | 73.8 µs | 43.2 µs | 4.7 µs | 15.0 µs | **0.59x** | **0.064x** | **0.20x** |
| **Matmul 64x64** | 46.8 µs | 26.3 µs | 5.0 µs | 14.8 µs | **0.56x** | **0.11x** | **0.32x** |
| **Matmul 256x256** | 44.0 µs | 923.9 µs | 252.6 µs | 760.5 µs | **21.00x** | **5.74x** | **17.28x** |
| **Cumsum** (256x256 over axis 1) | 65.1 µs | 30.7 µs | 117.2 µs | 138.1 µs | **0.47x** | **0.56x** | **0.47x** |
| **Matrix Power** (64x64 exponent 5) | 255.4 µs | 96.7 µs | 18.2 µs | 14.4 µs | **0.38x** | **0.071x** | **0.056x** |
| **Kronecker Product** (64x64 ⊗ 8x8) | 52.9 µs | 258.0 µs | — | 468.3 µs | **4.88x** | — | **8.85x** |
| **Dot Product** ($N = 65,536$) | 111.7 µs | 4.1 µs | 6.1 µs | — | **0.037x** | **0.055x** | — |
| **Trace** (256x256) | 88.1 µs | 136 ns | 200 ns | — | **0.0015x** | **0.0023x** | — |
| **Matrix Rank** (64x64 diagonal rank 32) | 6.46 ms | 150.2 µs | — | — | **0.023x** | — | — |
| **Determinant** (64x64 diagonal) | 8.42 ms | 71.4 µs | 38 ns | 9.0 µs | **0.0085x** | **0.0000045x** | **0.0011x** |
| **Blocked Cholesky Decomposition** (128x128 SPD) | 819.3 µs | 182.9 µs | — | 28.0 µs | **0.22x** | — | **0.034x** |
| **LU Decomposition** (32x32) | 150.7 µs | 12.7 µs | — | 1.9 µs | **0.084x** | — | **0.013x** |
| **Blocked LU Decomposition** (66x66) | 283.9 µs | 68.2 µs | — | 7.2 µs | **0.24x** | — | **0.025x** |
| **Full-Pivot LU Decomposition** (32x32) | 169.5 µs | 20.3 µs | — | 14.1 µs | **0.12x** | — | **0.083x** |
| **QR Decomposition** (48x24) | 134.9 µs | 11.5 µs | — | 4.3 µs | **0.085x** | — | **0.032x** |
| **Blocked QR Decomposition** (70x35) | 1.33 ms | 10.6 µs | — | 6.1 µs | **0.0079x** | — | **0.0046x** |
| **SVD Decomposition** (32x16) | 202.3 µs | 29.3 µs | — | 6.3 µs | **0.14x** | — | **0.031x** |
| **Bidiagonalization** (32x16) | 202.8 µs | 25.6 µs | — | 9.7 µs (nalgebra SVD) | **0.13x** | — | **0.048x** |
| **Schur Decomposition** (32x32) | 194.1 µs | 31.9 µs | — | 6.8 µs (nalgebra eigenvalues) | **0.16x** | — | **0.035x** |
| **Hessenberg Reduction** (32x32) | 211.1 µs | 45.9 µs | — | 6.7 µs | **0.22x** | — | **0.032x** |
| **Bunch-Kaufman Decomposition** (32x32) | 99.0 µs | 10.7 µs | — | 2.1 µs (nalgebra determinant) | **0.11x** | — | **0.021x** |
| **UDU Decomposition** (32x32) | 101.7 µs | 11.8 µs | — | 1.8 µs (nalgebra determinant) | **0.12x** | — | **0.017x** |
| **Symmetric Eigen Jacobi** (32x32) | 564.9 µs | 380.6 µs | — | 27.9 µs | **0.67x** | — | **0.049x** |
| **General Eigenvalues** (32x32 block rotations) | 169.2 µs | 25.9 µs | — | 6.7 µs | **0.15x** | — | **0.040x** |
| **Norm L1** ($N = 65,536$) | 139.3 µs | 2.6 µs | — | — | **0.019x** | — | — |
| **Norm L2** ($N = 65,536$) | 224.0 µs | 3.0 µs | — | — | **0.013x** | — | — |
| **Norm Max** ($N = 65,536$) | 146.0 µs | 2.6 µs | — | — | **0.018x** | — | — |
| **Column-Pivoted QR Decomposition** (32x32) | 104.2 µs | 26.3 µs | — | 14.5 µs | **0.25x** | — | **0.14x** |
| **Pseudoinverse** (32x32) | 1.92 ms | 1.78 ms | — | 19.6 µs | **0.93x** | — | **0.010x** |
| **Matrix Exponential** (32x32) | 347.1 µs | 196.0 µs | — | — | **0.56x** | — | — |

## Synchronization Profile

| Profile | Measured floor |
| --- | --- |
| **Blocked LU 66x66 transfer/synchronization floor** | 308.2 µs |
| **Blocked QR 70x35 transfer/synchronization floor** | 227.8 µs |

## Analysis

1. **Compute vs. Memory Bandwidth & GPU Scaling**:
   - For **Elementwise Add** (memory-bound, low arithmetic intensity), the GPU reaches $\approx 79.9 \text{ µs/iter}$, outperforming Leto by $\approx 12.86\times$ and `ndarray` by $\approx 13.14\times$ on this run.
   - For **Elementwise Exp** (compute-bound, high arithmetic intensity), the GPU reaches $\approx 45.8 \text{ µs/iter}$, outperforming Leto by $\approx 40.55\times$ and `ndarray` by $\approx 40.74\times$.
   - For **Matmul 256x256**, the GPU reaches $\approx 44.0 \text{ µs/iter}$ due to optimized parallel matrix contraction tiles, achieving **21.00x** speedup over Leto CPU and **17.28x** over nalgebra CPU.

2. **Driver Overhead and Small Input Regimes**:
   - For smaller workloads (e.g. **Trace**, **Dot Product**, **Norm L2**, and `ndarray` / `nalgebra` vector operations), the CPU dominates due to zero launch overhead and mature CPU kernels.
   - GPU operations require scheduling command buffers, copying layout metadata to uniform buffers, and dispatching to the GPU queue. Each dispatch incurs driver/runtime overhead.
   - The blocked decomposition rows show the current hybrid strategy is still
     synchronization-bound at these sizes: **Blocked LU 66x66** and
     **Blocked QR 70x35** trail both Leto and `nalgebra` despite GPU trailing
     updates. The next optimization target is reducing host/device round trips
     before adding more native decomposition kernels.
   - The synchronization profile attributes the 66x66 blocked LU row primarily
     to transfer/synchronization cost. For 70x35 blocked QR, transfer cost is
     material, but the gap also includes per-reflector kernel launches and
     vector uploads.
