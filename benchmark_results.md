# hephaestus comparative CPU/GPU baselines

Harness: `crates/hephaestus-wgpu/benches/comparative.rs` (`cargo bench --bench comparative -p hephaestus-wgpu`).
Methodology: 50 iterations, wall-time divided by iteration count, including GPU synchronization (`poll(wgpu::PollType::Wait)`) on the host side.
Inputs: Contiguous `f32` vectors/matrices of varying shapes (scaled to prevent overflow).
Machine Class: Windows 11 x86_64 dev workstation.

## Comparative Results

| Benchmark | GPU (WGPU) | Leto CPU | ndarray CPU | nalgebra CPU | GPU Speedup (vs Leto) | GPU Speedup (vs ndarray) | GPU Speedup (vs nalgebra) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Elementwise Add** ($N = 2^{20}$) | 56.3 µs | 1.19 ms | 1.19 ms | — | **21.2x** | **21.1x** | — |
| **Elementwise Exp** ($N = 2^{20}$) | 34.2 µs | 1.91 ms | 1.97 ms | — | **55.9x** | **57.5x** | — |
| **Sum Reduction** ($N = 2^{20}$) | 42.9 µs | 61.6 µs | 96.8 µs | — | **1.44x** | **2.26x** | — |
| **Axis Sum** (256x256 over axis 0) | 65.7 µs | 38.6 µs | 4.1 µs | 16.1 µs | **0.59x** | **0.062x** | **0.25x** |
| **Axis Min** (256x256 over axis 0) | 60.6 µs | 57.6 µs | 8.9 µs | 12.1 µs | **0.95x** | **0.15x** | **0.20x** |
| **Axis Max** (256x256 over axis 0) | 63.9 µs | 44.3 µs | 7.1 µs | 12.3 µs | **0.69x** | **0.11x** | **0.19x** |
| **Axis Mean** (256x256 over axis 0) | 61.9 µs | 42.6 µs | 4.9 µs | 21.1 µs | **0.69x** | **0.079x** | **0.34x** |
| **Matmul 64x64** | 38.8 µs | 11.9 µs | 14.1 µs | 17.9 µs | **0.31x** | **0.36x** | **0.46x** |
| **Matmul 256x256** | 40.5 µs | 667.3 µs | 374.9 µs | 1.28 ms | **16.5x** | **9.26x** | **31.6x** |
| **Cumsum** (256x256 over axis 1) | 31.4 µs | 32.0 µs | 112.5 µs | 137.1 µs | **1.02x** | **3.59x** | **4.37x** |
| **Matrix Power** (64x64 exponent 5) | 129.1 µs | 40.0 µs | 19.3 µs | 16.6 µs | **0.31x** | **0.15x** | **0.13x** |
| **Kronecker Product** (64x64 ⊗ 8x8) | 28.6 µs | 231.3 µs | — | 409.0 µs | **8.09x** | — | **14.3x** |
| **Dot Product** ($N = 65,536$) | 57.2 µs | 3.7 µs | 7.7 µs | — | **0.066x** | **0.14x** | — |
| **Trace** (256x256) | 26.2 µs | 122 ns | 152 ns | — | **0.0047x** | **0.0058x** | — |
| **Matrix Rank** (64x64 diagonal rank 32) | 4.74 ms | 106.8 µs | — | — | **0.023x** | — | — |
| **Determinant** (64x64 diagonal) | 6.16 ms | 61.3 µs | 20 ns | 6.8 µs | **0.0099x** | **0.0000032x** | **0.0011x** |
| **Blocked Cholesky Decomposition** (128x128 SPD) | 530.0 µs | 149.5 µs | — | 30.5 µs | **0.28x** | — | **0.058x** |
| **LU Decomposition** (32x32) | 83.1 µs | 9.4 µs | — | 1.5 µs | **0.11x** | — | **0.018x** |
| **QR Decomposition** (48x24) | 86.8 µs | 14.0 µs | — | 2.8 µs | **0.16x** | — | **0.033x** |
| **Norm L1** ($N = 65,536$) | 62.8 µs | 1.7 µs | — | — | **0.027x** | — | — |
| **Norm L2** ($N = 65,536$) | 120.2 µs | 2.5 µs | — | — | **0.021x** | — | — |
| **Norm Max** ($N = 65,536$) | 64.6 µs | 2.1 µs | — | — | **0.033x** | — | — |

## Analysis

1. **Compute vs. Memory Bandwidth & GPU Scaling**:
   - For **Elementwise Add** (memory-bound, low arithmetic intensity), the GPU reaches $\approx 56.3 \text{ µs/iter}$, outperforming Leto by $\approx 21.2\times$ and `ndarray` by $\approx 21.1\times$ on this run.
   - For **Elementwise Exp** (compute-bound, high arithmetic intensity), the GPU reaches $\approx 34.2 \text{ µs/iter}$, outperforming Leto by $\approx 55.9\times$ and `ndarray` by $\approx 57.5\times$.
   - For **Axis Sum/Min/Max/Mean** (256x256 over axis 0), the GPU trails Leto, `ndarray`, and the nalgebra-backed references at this size.
   - For **Cumsum** (256x256 over axis 1), the GPU is at parity with Leto ($\approx 1.02\times$) and faster than the scalar ndarray/nalgebra-backed references in this harness.
   - For **Matmul 256x256** ($1.67 \times 10^7$ multiply-accumulates), the GPU is $\approx 16.5\times$ faster than Leto CPU, $\approx 9.26\times$ faster than `ndarray`, and $\approx 31.6\times$ faster than `nalgebra` in this harness. This shows the effectiveness of the tiled WGSL matrix multiplication implementation utilizing shared memory barriers.
   - For **Matrix Power** (64x64 exponent 5), the GPU trails Leto, `ndarray`, and `nalgebra` for this small matrix because the implementation performs four separate GPU matmul dispatches plus identity/copy setup.
   - For **Kronecker Product** (64x64 ⊗ 8x8, 262,144 output elements), the GPU reaches $\approx 28.6 \text{ µs/iter}$, outperforming Leto by $\approx 8.09\times$ and the nalgebra-backed reference by $\approx 14.3\times$.
   - For **Matrix Rank** (64x64 diagonal rank 32), the current GPU row-reduction kernel is $\approx 44\times$ slower than Leto's SVD-based rank because the WGPU implementation intentionally uses one invocation for correctness-first serial pivoting. This is an optimization gap, not a benchmark threshold to weaken.
   - For **Determinant** (64x64 diagonal), the GPU row-reduction determinant is $\approx 101\times$ slower than Leto and much slower than `ndarray`'s diagonal-product reference. This is correctness/parity coverage for the matrix-property API, not an optimization claim.
   - For **Blocked Cholesky** (128x128 SPD), WGPU runs a CPU panel factorization/solve plus GPU SYRK trailing update and reaches $\approx 530.0 \text{ µs/iter}$ versus $\approx 149.5 \text{ µs/iter}$ for Leto and $\approx 30.5 \text{ µs/iter}$ for nalgebra. This proves the GPU trailing-update path and quantifies current overhead; it is not a performance win yet.
   - For **LU** and **QR**, the current WGPU API stores factors on the device but delegates factorization to Leto on the host, so GPU timings include transfer and host factorization overhead. These rows prove API parity and quantify the current cost; they do not prove GPU-kernel decomposition performance.

2. **Driver Overhead and Small Input Regimes**:
   - For smaller workloads (e.g. **Trace**, **Dot Product**, **Norm L2**, and `ndarray` / `nalgebra` vector operations at $N=65,536$), the CPU dominates due to zero launch overhead and mature CPU kernels.
   - GPU operations require scheduling command buffers, copying layout metadata to uniform buffers, and dispatching to the GPU queue. Each dispatch incurs a drivers/runtime overhead of $\approx 20\text{--}30 \text{ µs}$.
   - **Trace** and **Norm L1** use fused map-reduction kernels in this run. Dot product, L2 norm, and max norm retain their two-stage map-then-reduce paths because the fused variant regressed in the local empirical run.
   - For composite operations like **Dot Product** (Elementwise Multiply + Sum Reduction) and **Norm L2** (Square + Sum Reduction + Sqrt), multiple dispatches dominate the runtime at $\approx 55\text{--}59 \text{ µs}$.
   - Consequently, for datasets of size $N=65,536$ (only 256KB), low-arithmetic-density operations often finish on the CPU before the GPU dispatch sequence amortizes. GPU benefit requires larger data volume, higher arithmetic density, or fused kernels that reduce dispatch count.

3. **Reduction Efficiency**:
   - The **Sum Reduction** benchmark shows the GPU is $\approx 1.44\times$ faster than Leto and $\approx 2.26\times$ faster than `ndarray`. WGPU utilizes a multi-pass tree reduction kernel with shared workgroup memory, which minimizes the overhead of global memory transactions.
