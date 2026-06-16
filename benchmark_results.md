# hephaestus comparative CPU/GPU baselines

Harness: `crates/hephaestus-wgpu/benches/comparative.rs` (`cargo bench --bench comparative -p hephaestus-wgpu`).
Methodology: 50 iterations, wall-time divided by iteration count, including GPU synchronization (`poll(wgpu::PollType::Wait)`) on the host side.
Inputs: Contiguous `f32` vectors/matrices of varying shapes (scaled to prevent overflow).
Machine Class: Windows 11 x86_64 dev workstation.

## Comparative Results

| Benchmark | GPU (WGPU) | Leto CPU | ndarray CPU | nalgebra CPU | GPU Speedup (vs Leto) | GPU Speedup (vs ndarray) | GPU Speedup (vs nalgebra) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Elementwise Add** ($N = 2^{20}$) | 93.1 µs | 1.19 ms | 1.22 ms | — | **12.8x** | **13.1x** | — |
| **Elementwise Exp** ($N = 2^{20}$) | 24.7 µs | 2.11 ms | 2.02 ms | — | **85.5x** | **81.9x** | — |
| **Sum Reduction** ($N = 2^{20}$) | 44.6 µs | 72.6 µs | 80.4 µs | — | **1.6x** | **1.8x** | — |
| **Axis Sum** (256x256 over axis 0) | 63.5 µs | 284.5 µs | 3.4 µs | 15.2 µs | **4.5x** | **0.054x** | **0.24x** |
| **Axis Min** (256x256 over axis 0) | 64.7 µs | 276.8 µs | 3.9 µs | 5.2 µs | **4.3x** | **0.060x** | **0.081x** |
| **Axis Max** (256x256 over axis 0) | 64.7 µs | 286.4 µs | 3.9 µs | 5.0 µs | **4.4x** | **0.060x** | **0.078x** |
| **Axis Mean** (256x256 over axis 0) | 71.9 µs | 280.9 µs | 3.7 µs | 15.5 µs | **3.9x** | **0.052x** | **0.22x** |
| **Matmul 64x64** | 23.7 µs | 16.5 µs | 4.8 µs | 4.6 µs | **0.70x** | **0.20x** | **0.19x** |
| **Matmul 256x256** | 37.7 µs | 1.63 ms | 250.5 µs | 245.4 µs | **43.1x** | **6.6x** | **6.5x** |
| **Cumsum** (256x256 over axis 1) | 32.5 µs | 31.0 µs | 111.9 µs | 142.2 µs | **0.95x** | **3.4x** | **4.4x** |
| **Matrix Power** (64x64 exponent 5) | 127.2 µs | 65.3 µs | 18.1 µs | 14.5 µs | **0.51x** | **0.14x** | **0.11x** |
| **Kronecker Product** (64x64 ⊗ 8x8) | 29.3 µs | 1.39 ms | 423.7 µs | 415.4 µs | **47.3x** | **14.4x** | **14.2x** |
| **Dot Product** ($N = 65,536$) | 55.6 µs | 3.8 µs | 6.3 µs | 7.8 µs | **0.069x** | **0.11x** | **0.14x** |
| **Trace** (256x256) | 26.5 µs | 1.1 µs | 80 ns | 82 ns | **0.042x** | **0.0030x** | **0.0031x** |
| **Norm L1** ($N = 65,536$) | 84.3 µs | 1.7 µs | 9.4 µs | 51.3 µs | **0.020x** | **0.11x** | **0.61x** |
| **Norm L2** ($N = 65,536$) | 66.6 µs | 2.4 µs | 8.3 µs | 4.2 µs | **0.036x** | **0.13x** | **0.062x** |
| **Norm Max** ($N = 65,536$) | 57.7 µs | 2.2 µs | 4.5 µs | 124.1 µs | **0.038x** | **0.079x** | **2.2x** |

## Analysis

1. **Compute vs. Memory Bandwidth & GPU Scaling**:
   - For **Elementwise Add** (memory-bound, low arithmetic intensity), the GPU reaches $\approx 93.1 \text{ µs/iter}$, outperforming Leto by $\approx 12.8\times$ and `ndarray` by $\approx 13.1\times$ on this run.
   - For **Elementwise Exp** (compute-bound, high arithmetic intensity), the GPU reaches $\approx 24.7 \text{ µs/iter}$, outperforming Leto by $\approx 85.5\times$ and `ndarray` by $\approx 81.9\times$.
   - For **Axis Sum/Min/Max/Mean** (256x256 over axis 0), the GPU outperforms Leto by $\approx 4.5\times$, $\approx 4.3\times$, $\approx 4.4\times$, and $\approx 3.9\times$ respectively, while trailing `ndarray` and the nalgebra-backed references at this size.
   - For **Cumsum** (256x256 over axis 1), the GPU is roughly at parity with Leto ($\approx 0.95\times$) and faster than the scalar ndarray/nalgebra-backed references in this harness.
   - For **Matmul 256x256** ($1.67 \times 10^7$ multiply-accumulates), the GPU is $\approx 43\times$ faster than Leto CPU and $\approx 6.5\times$ faster than `ndarray` and `nalgebra` in this harness. This shows the effectiveness of the tiled WGSL matrix multiplication implementation utilizing shared memory barriers.
   - For **Matrix Power** (64x64 exponent 5), the GPU trails Leto, `ndarray`, and `nalgebra` for this small matrix because the implementation performs four separate GPU matmul dispatches plus identity/copy setup.
   - For **Kronecker Product** (64x64 ⊗ 8x8, 262,144 output elements), the GPU reaches $\approx 29.3 \text{ µs/iter}$, outperforming Leto by $\approx 47.3\times$, `ndarray` by $\approx 14.4\times$, and the nalgebra-backed reference by $\approx 14.2\times$.

2. **Driver Overhead and Small Input Regimes**:
   - For smaller workloads (e.g. **Trace**, **Dot Product**, **Norm L2**, and `ndarray` / `nalgebra` vector operations at $N=65,536$), the CPU dominates due to zero launch overhead and mature CPU kernels.
   - GPU operations require scheduling command buffers, copying layout metadata to uniform buffers, and dispatching to the GPU queue. Each dispatch incurs a drivers/runtime overhead of $\approx 20\text{--}30 \text{ µs}$.
   - **Trace** and **Norm L1** use fused map-reduction kernels in this run. Dot product, L2 norm, and max norm retain their two-stage map-then-reduce paths because the fused variant regressed in the local empirical run.
   - For composite operations like **Dot Product** (Elementwise Multiply + Sum Reduction) and **Norm L2** (Square + Sum Reduction + Sqrt), multiple dispatches dominate the runtime at $\approx 56\text{--}67 \text{ µs}$.
   - Consequently, for datasets of size $N=65,536$ (only 256KB), low-arithmetic-density operations often finish on the CPU before the GPU dispatch sequence amortizes. GPU benefit requires larger data volume, higher arithmetic density, or fused kernels that reduce dispatch count.

3. **Reduction Efficiency**:
   - The **Sum Reduction** benchmark shows the GPU is $\approx 1.6\times$ faster than Leto and $\approx 1.8\times$ faster than `ndarray`. WGPU utilizes a multi-pass tree reduction kernel with shared workgroup memory, which minimizes the overhead of global memory transactions.
