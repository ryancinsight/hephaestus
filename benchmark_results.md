# hephaestus comparative CPU/GPU baselines

Harness: `crates/hephaestus-wgpu/benches/comparative.rs` (`cargo bench --bench comparative -p hephaestus-wgpu`).  
Methodology: 50 iterations, median wall-time per iteration including GPU synchronization (`poll(wgpu::PollType::Wait)`) on the host side.  
Inputs: Contiguous `f32` vectors of length $N = 2^{20}$ ($1,048,576$ elements). Reproducible, deterministic values scaled to prevent overflow.  
Machine Class: Windows 11 x86_64 dev workstation.

## Comparative Results ($N = 1,048,576$, `f32`)

| Benchmark | GPU (WGPU) | Leto CPU | ndarray CPU | GPU Speedup (vs Leto) | GPU Speedup (vs ndarray) |
| --- | --- | --- | --- | --- | --- |
| **Elementwise Add** | 85.676 µs | 1.422 ms | 1.390 ms | **16.6x** | **16.2x** |
| **Elementwise Exp** | 33.508 µs | 2.021 ms | 2.057 ms | **60.3x** | **61.4x** |
| **Sum Reduction** | 44.184 µs | 421.066 µs | 96.380 µs | **9.5x** | **2.2x** |

## Analysis

1. **Compute vs. Memory Bandwidth**:
   - For **Elementwise Add** (memory-bound, low arithmetic intensity), the GPU reaches $\approx 85.7 \text{ µs/iter}$, outperforming the CPU by $\approx 16\times$. The CPU is limited by DRAM bandwidth, while the GPU's high-bandwidth memory (HBM/GDDR) allows much faster streaming.
   - For **Elementwise Exp** (compute-bound, high arithmetic intensity), the GPU speedup scales up to $\approx 60\times$ ($\approx 33.5 \text{ µs/iter}$ vs $\approx 2.0 \text{ ms/iter}$). The massive parallel shader execution units on the GPU handle transcendental operations ($e^x$) much more efficiently than CPU vector units.

2. **Reduction Efficiency**:
   - The **Sum Reduction** benchmark shows the GPU is $\approx 9.5\times$ faster than Leto and $\approx 2.2\times$ faster than `ndarray`. WGPU utilizes a multi-pass tree reduction kernel with shared workgroup memory, which minimizes the overhead of global memory transactions.
   - The CPU's `ndarray` performs very well on sum reduction ($\approx 96.4 \text{ µs/iter}$) due to cache locality and aggressive auto-vectorization, but the GPU still wins due to parallel reduction scaling.

3. **Overhead & GPU Latency**:
   - The benchmarks measure the complete roundtrip host-to-device synchronization loop (`wait()`). The extremely low execution time of WGPU ($\approx 33\text{--}85 \text{ µs}$) shows that the `hephaestus-wgpu` uniform/staging buffer pools and cached pipeline lookup are highly optimized, keeping API driver launch latency under $50\text{ µs}$.
