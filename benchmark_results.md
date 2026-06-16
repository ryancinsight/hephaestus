# hephaestus comparative CPU/GPU baselines

Harness: `crates/hephaestus-wgpu/benches/comparative.rs` (`cargo bench --bench comparative -p hephaestus-wgpu`).
Methodology: 50 iterations, wall-time divided by iteration count, including GPU synchronization (`poll(wgpu::PollType::Wait)`) on the host side.
Inputs: Contiguous `f32` vectors/matrices of varying shapes (scaled to prevent overflow).
Machine Class: Windows 11 x86_64 dev workstation.

## Comparative Results

| Benchmark | GPU (WGPU) | Leto CPU | ndarray CPU | nalgebra CPU | GPU Speedup (vs Leto) | GPU Speedup (vs ndarray) | GPU Speedup (vs nalgebra) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Elementwise Add** ($N = 2^{20}$) | 44.7 µs | 1.00 ms | 1.03 ms | — | **22.4x** | **23.0x** | — |
| **Elementwise Exp** ($N = 2^{20}$) | 33.0 µs | 1.88 ms | 1.85 ms | — | **56.9x** | **56.3x** | — |
| **Sum Reduction** ($N = 2^{20}$) | 41.9 µs | 71.2 µs | 83.4 µs | — | **1.70x** | **1.99x** | — |
| **Axis Sum** (256x256 over axis 0) | 58.6 µs | 41.8 µs | 3.4 µs | 15.2 µs | **0.71x** | **0.058x** | **0.26x** |
| **Axis Min** (256x256 over axis 0) | 60.0 µs | 47.6 µs | 4.2 µs | 5.5 µs | **0.79x** | **0.070x** | **0.091x** |
| **Axis Max** (256x256 over axis 0) | 59.1 µs | 48.9 µs | 3.6 µs | 5.0 µs | **0.83x** | **0.062x** | **0.085x** |
| **Axis Mean** (256x256 over axis 0) | 62.3 µs | 42.1 µs | 3.1 µs | 15.2 µs | **0.68x** | **0.050x** | **0.24x** |
| **Matmul 64x64** | 41.9 µs | 10.3 µs | 4.7 µs | 18.1 µs | **0.25x** | **0.11x** | **0.43x** |
| **Matmul 256x256** | 38.2 µs | 662.9 µs | 280.7 µs | 1.01 ms | **17.4x** | **7.4x** | **26.5x** |
| **Cumsum** (256x256 over axis 1) | 39.5 µs | 30.5 µs | 108.9 µs | 156.7 µs | **0.77x** | **2.8x** | **4.0x** |
| **Matrix Power** (64x64 exponent 5) | 124.9 µs | 41.0 µs | 18.2 µs | 14.2 µs | **0.33x** | **0.15x** | **0.11x** |
| **Kronecker Product** (64x64 ⊗ 8x8) | 26.6 µs | 218.4 µs | — | 420.6 µs | **8.2x** | — | **15.8x** |
| **Dot Product** ($N = 65,536$) | 55.6 µs | 3.3 µs | 6.2 µs | — | **0.059x** | **0.11x** | — |
| **Trace** (256x256) | 28.2 µs | 124 ns | 160 ns | — | **0.0044x** | **0.0057x** | — |
| **Matrix Rank** (64x64 diagonal rank 32) | 4.65 ms | 112.1 µs | — | — | **0.024x** | — | — |
| **Determinant** (64x64 diagonal) | 6.09 ms | 57.3 µs | 24 ns | 6.8 µs | **0.0094x** | **0.0000039x** | **0.0011x** |
| **Norm L1** ($N = 65,536$) | 35.6 µs | 2.2 µs | — | — | **0.062x** | — | — |
| **Norm L2** ($N = 65,536$) | 61.3 µs | 2.9 µs | — | — | **0.048x** | — | — |
| **Norm Max** ($N = 65,536$) | 52.3 µs | 2.2 µs | — | — | **0.042x** | — | — |

## Analysis

1. **Compute vs. Memory Bandwidth & GPU Scaling**:
   - For **Elementwise Add** (memory-bound, low arithmetic intensity), the GPU reaches $\approx 44.7 \text{ µs/iter}$, outperforming Leto by $\approx 22.4\times$ and `ndarray` by $\approx 23.0\times$ on this run.
   - For **Elementwise Exp** (compute-bound, high arithmetic intensity), the GPU reaches $\approx 33.0 \text{ µs/iter}$, outperforming Leto by $\approx 56.9\times$ and `ndarray` by $\approx 56.3\times$.
   - For **Axis Sum/Min/Max/Mean** (256x256 over axis 0), the GPU trails Leto, `ndarray`, and the nalgebra-backed references at this size.
   - For **Cumsum** (256x256 over axis 1), the GPU is roughly at parity with Leto ($\approx 0.95\times$) and faster than the scalar ndarray/nalgebra-backed references in this harness.
   - For **Matmul 256x256** ($1.67 \times 10^7$ multiply-accumulates), the GPU is $\approx 17.4\times$ faster than Leto CPU, $\approx 7.4\times$ faster than `ndarray`, and $\approx 26.5\times$ faster than `nalgebra` in this harness. This shows the effectiveness of the tiled WGSL matrix multiplication implementation utilizing shared memory barriers.
   - For **Matrix Power** (64x64 exponent 5), the GPU trails Leto, `ndarray`, and `nalgebra` for this small matrix because the implementation performs four separate GPU matmul dispatches plus identity/copy setup.
   - For **Kronecker Product** (64x64 ⊗ 8x8, 262,144 output elements), the GPU reaches $\approx 26.6 \text{ µs/iter}$, outperforming Leto by $\approx 8.2\times$ and the nalgebra-backed reference by $\approx 15.8\times$.
   - For **Matrix Rank** (64x64 diagonal rank 32), the current GPU row-reduction kernel is $\approx 42\times$ slower than Leto's SVD-based rank because the WGPU implementation intentionally uses one invocation for correctness-first serial pivoting. This is an optimization gap, not a benchmark threshold to weaken.
   - For **Determinant** (64x64 diagonal), the GPU row-reduction determinant is $\approx 106\times$ slower than Leto and much slower than `ndarray`'s diagonal-product reference. This is correctness/parity coverage for the matrix-property API, not an optimization claim.

2. **Driver Overhead and Small Input Regimes**:
   - For smaller workloads (e.g. **Trace**, **Dot Product**, **Norm L2**, and `ndarray` / `nalgebra` vector operations at $N=65,536$), the CPU dominates due to zero launch overhead and mature CPU kernels.
   - GPU operations require scheduling command buffers, copying layout metadata to uniform buffers, and dispatching to the GPU queue. Each dispatch incurs a drivers/runtime overhead of $\approx 20\text{--}30 \text{ µs}$.
   - **Trace** and **Norm L1** use fused map-reduction kernels in this run. Dot product, L2 norm, and max norm retain their two-stage map-then-reduce paths because the fused variant regressed in the local empirical run.
   - For composite operations like **Dot Product** (Elementwise Multiply + Sum Reduction) and **Norm L2** (Square + Sum Reduction + Sqrt), multiple dispatches dominate the runtime at $\approx 55\text{--}59 \text{ µs}$.
   - Consequently, for datasets of size $N=65,536$ (only 256KB), low-arithmetic-density operations often finish on the CPU before the GPU dispatch sequence amortizes. GPU benefit requires larger data volume, higher arithmetic density, or fused kernels that reduce dispatch count.

3. **Reduction Efficiency**:
   - The **Sum Reduction** benchmark shows the GPU is $\approx 1.70\times$ faster than Leto and $\approx 1.99\times$ faster than `ndarray`. WGPU utilizes a multi-pass tree reduction kernel with shared workgroup memory, which minimizes the overhead of global memory transactions.
