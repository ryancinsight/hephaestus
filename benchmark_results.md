# hephaestus comparative CPU/GPU baselines

Harness: `crates/hephaestus-wgpu/benches/comparative.rs` (`cargo bench --bench comparative -p hephaestus-wgpu`).
Methodology: 50 iterations, wall-time divided by iteration count, including GPU synchronization (`poll(wgpu::PollType::Wait)`) on the host side.
Inputs: Contiguous `f32` vectors/matrices of varying shapes (scaled to prevent overflow).
Machine Class: Windows 11 x86_64 dev workstation.

## Comparative Results

| Benchmark | GPU (WGPU) | Leto CPU | ndarray CPU | nalgebra CPU | GPU Speedup (vs Leto) | GPU Speedup (vs ndarray) | GPU Speedup (vs nalgebra) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Elementwise Add** ($N = 2^{20}$) | 41.7 µs | 1.35 ms | 1.38 ms | — | **32.3x** | **33.2x** | — |
| **Elementwise Exp** ($N = 2^{20}$) | 47.1 µs | 2.23 ms | 2.13 ms | — | **47.4x** | **45.3x** | — |
| **Sum Reduction** ($N = 2^{20}$) | 43.2 µs | 57.0 µs | 79.1 µs | — | **1.32x** | **1.83x** | — |
| **Axis Sum** (256x256 over axis 0) | 58.8 µs | 42.8 µs | 3.6 µs | 15.5 µs | **0.73x** | **0.060x** | **0.26x** |
| **Axis Min** (256x256 over axis 0) | 59.0 µs | 47.7 µs | 4.0 µs | 5.7 µs | **0.81x** | **0.068x** | **0.096x** |
| **Axis Max** (256x256 over axis 0) | 58.6 µs | 47.5 µs | 4.1 µs | 5.6 µs | **0.81x** | **0.070x** | **0.096x** |
| **Axis Mean** (256x256 over axis 0) | 63.8 µs | 42.6 µs | 3.7 µs | 15.1 µs | **0.67x** | **0.058x** | **0.24x** |
| **Matmul 64x64** | 22.9 µs | 10.2 µs | 5.0 µs | 14.7 µs | **0.45x** | **0.22x** | **0.64x** |
| **Matmul 256x256** | 37.3 µs | 647.2 µs | 409.1 µs | 1.14 ms | **17.3x** | **11.0x** | **30.6x** |
| **Cumsum** (256x256 over axis 1) | 33.4 µs | 30.4 µs | 176.3 µs | 147.6 µs | **0.91x** | **5.28x** | **4.42x** |
| **Matrix Power** (64x64 exponent 5) | 160.9 µs | 42.7 µs | 18.6 µs | 14.7 µs | **0.27x** | **0.12x** | **0.091x** |
| **Kronecker Product** (64x64 ⊗ 8x8) | 26.8 µs | 266.5 µs | — | 479.6 µs | **9.94x** | — | **17.9x** |
| **Dot Product** ($N = 65,536$) | 59.2 µs | 3.3 µs | 7.7 µs | — | **0.055x** | **0.13x** | — |
| **Trace** (256x256) | 25.6 µs | 122 ns | 164 ns | — | **0.0048x** | **0.0064x** | — |
| **Matrix Rank** (64x64 diagonal rank 32) | 4.61 ms | 101.6 µs | — | — | **0.022x** | — | — |
| **Determinant** (64x64 diagonal) | 6.04 ms | 61.0 µs | 22 ns | 7.0 µs | **0.010x** | **0.0000036x** | **0.0012x** |
| **Blocked Cholesky Decomposition** (128x128 SPD) | 517.2 µs | 149.9 µs | — | 26.4 µs | **0.29x** | — | **0.051x** |
| **LU Decomposition** (32x32) | 89.3 µs | 11.6 µs | — | 1.5 µs | **0.13x** | — | **0.017x** |
| **QR Decomposition** (48x24) | 84.2 µs | 14.7 µs | — | 2.7 µs | **0.17x** | — | **0.032x** |
| **Symmetric Eigen Jacobi** (32x32) | 379.7 µs | 303.0 µs | — | 21.1 µs | **0.80x** | — | **0.056x** |
| **Norm L1** ($N = 65,536$) | 47.8 µs | 1.8 µs | — | — | **0.037x** | — | — |
| **Norm L2** ($N = 65,536$) | 62.0 µs | 2.4 µs | — | — | **0.039x** | — | — |
| **Norm Max** ($N = 65,536$) | 53.2 µs | 2.1 µs | — | — | **0.040x** | — | — |

## Analysis

1. **Compute vs. Memory Bandwidth & GPU Scaling**:
   - For **Elementwise Add** (memory-bound, low arithmetic intensity), the GPU reaches $\approx 41.7 \text{ µs/iter}$, outperforming Leto by $\approx 32.3\times$ and `ndarray` by $\approx 33.2\times$ on this run.
   - For **Elementwise Exp** (compute-bound, high arithmetic intensity), the GPU reaches $\approx 47.1 \text{ µs/iter}$, outperforming Leto by $\approx 47.4\times$ and `ndarray` by $\approx 45.3\times$.
   - For **Axis Sum/Min/Max/Mean** (256x256 over axis 0), the GPU trails Leto, `ndarray`, and the nalgebra-backed references at this size.
   - For **Cumsum** (256x256 over axis 1), the GPU trails Leto on this run but remains faster than the scalar ndarray/nalgebra-backed references in this harness.
   - For **Matmul 256x256** ($1.67 \times 10^7$ multiply-accumulates), the GPU is $\approx 17.3\times$ faster than Leto CPU, $\approx 11.0\times$ faster than `ndarray`, and $\approx 30.6\times$ faster than `nalgebra` in this harness. This shows the effectiveness of the tiled WGSL matrix multiplication implementation utilizing shared memory barriers.
   - For **Matrix Power** (64x64 exponent 5), the GPU trails Leto, `ndarray`, and `nalgebra` for this small matrix because the implementation performs four separate GPU matmul dispatches plus identity/copy setup.
   - For **Kronecker Product** (64x64 ⊗ 8x8, 262,144 output elements), the GPU reaches $\approx 26.8 \text{ µs/iter}$, outperforming Leto by $\approx 9.94\times$ and the nalgebra-backed reference by $\approx 17.9\times$.
   - For **Matrix Rank** (64x64 diagonal rank 32), the current GPU row-reduction kernel is $\approx 45\times$ slower than Leto's SVD-based rank because the WGPU implementation intentionally uses one invocation for correctness-first serial pivoting. This is an optimization gap, not a benchmark threshold to weaken.
   - For **Determinant** (64x64 diagonal), the GPU row-reduction determinant is $\approx 99\times$ slower than Leto and much slower than `ndarray`'s diagonal-product reference. This is correctness/parity coverage for the matrix-property API, not an optimization claim.
   - For **Blocked Cholesky** (128x128 SPD), WGPU runs a CPU panel factorization/solve plus GPU SYRK trailing update and reaches $\approx 517.2 \text{ µs/iter}$ versus $\approx 149.9 \text{ µs/iter}$ for Leto and $\approx 26.4 \text{ µs/iter}$ for nalgebra. This proves the GPU trailing-update path and quantifies current overhead; it is not a performance win yet.
   - For **LU**, **QR**, and **Symmetric Eigen Jacobi**, the current WGPU API stores results on the device but delegates the decomposition/eigensolve to Leto on the host, so GPU timings include transfer and host algorithm overhead. These rows prove API parity and quantify the current cost; they do not prove GPU-kernel decomposition performance.

2. **Driver Overhead and Small Input Regimes**:
   - For smaller workloads (e.g. **Trace**, **Dot Product**, **Norm L2**, and `ndarray` / `nalgebra` vector operations at $N=65,536$), the CPU dominates due to zero launch overhead and mature CPU kernels.
   - GPU operations require scheduling command buffers, copying layout metadata to uniform buffers, and dispatching to the GPU queue. Each dispatch incurs a drivers/runtime overhead of $\approx 20\text{--}30 \text{ µs}$.
   - **Trace** and **Norm L1** use fused map-reduction kernels in this run. Dot product, L2 norm, and max norm retain their two-stage map-then-reduce paths because the fused variant regressed in the local empirical run.
   - For composite operations like **Dot Product** (Elementwise Multiply + Sum Reduction) and **Norm L2** (Square + Sum Reduction + Sqrt), multiple dispatches dominate the runtime at $\approx 59\text{--}62 \text{ µs}$.
   - Consequently, for datasets of size $N=65,536$ (only 256KB), low-arithmetic-density operations often finish on the CPU before the GPU dispatch sequence amortizes. GPU benefit requires larger data volume, higher arithmetic density, or fused kernels that reduce dispatch count.

3. **Reduction Efficiency**:
   - The **Sum Reduction** benchmark shows the GPU is $\approx 1.32\times$ faster than Leto and $\approx 1.83\times$ faster than `ndarray`. WGPU utilizes a multi-pass tree reduction kernel with shared workgroup memory, which minimizes the overhead of global memory transactions.
