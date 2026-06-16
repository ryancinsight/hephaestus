import time
import numpy as np
import pyhephaestus as ph

# Try importing cupy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

def run_bench():
    print("======================================================================")
    print("      Hephaestus (WGPU) vs NumPy (CPU) vs CuPy (GPU) Benchmark        ")
    print("======================================================================")
    print(f"CuPy available: {CUPY_AVAILABLE}\n")

    # Initialize device
    dev = ph.Device()
    print(f"Initialized Hephaestus Device (Backend: {dev.backend_name})")

    sizes = [1_000, 10_000, 100_000, 1_000_000, 5_000_000]

    for size in sizes:
        print(f"\n--- Array Size: {size:,} f32 elements ---")

        # 1. Correctness Validation
        # Generate random host data
        a_host = np.random.uniform(-1.0, 1.0, size).astype(np.float32)
        b_host = np.random.uniform(-1.0, 1.0, size).astype(np.float32)

        # Upload to Hephaestus
        a_ph = ph.Array(a_host.tolist(), dev)
        b_ph = ph.Array(b_host.tolist(), dev)

        # Add correctness
        add_ph = a_ph + b_ph
        add_ph_host = np.array(add_ph.tolist(), dtype=np.float32)
        add_np = a_host + b_host
        np.testing.assert_allclose(add_ph_host, add_np, rtol=1e-5, atol=1e-5)

        # Exp correctness
        exp_ph = a_ph.exp()
        exp_ph_host = np.array(exp_ph.tolist(), dtype=np.float32)
        exp_np = np.exp(a_host)
        np.testing.assert_allclose(exp_ph_host, exp_np, rtol=1e-5, atol=1e-5)

        # Scalar operations correctness
        scalar_add_ph = a_ph + 5.0
        scalar_add_ph_host = np.array(scalar_add_ph.tolist(), dtype=np.float32)
        scalar_add_np = a_host + 5.0
        np.testing.assert_allclose(scalar_add_ph_host, scalar_add_np, rtol=1e-5, atol=1e-5)

        # Reduction sum correctness
        sum_ph = a_ph.sum()
        sum_ph_val = sum_ph.tolist()[0]
        sum_np = np.sum(a_host)
        np.testing.assert_allclose(sum_ph_val, sum_np, rtol=1e-3, atol=1e-3)

        print("-> Correctness validation PASSED!")

        # 2. Performance benchmarking
        # NumPy
        t0 = time.perf_counter()
        for _ in range(50):
            res_np = a_host + b_host
            res_np = np.exp(res_np)
        t_np = (time.perf_counter() - t0) / 50.0

        # Hephaestus (WGPU)
        # Warmup
        for _ in range(5):
            res_ph = a_ph + b_ph
            res_ph = res_ph.exp()
        t0 = time.perf_counter()
        for _ in range(50):
            res_ph = a_ph + b_ph
            res_ph = res_ph.exp()
        # Wait for device queue to finish by downloading final result
        res_ph.tolist()
        t_ph = (time.perf_counter() - t0) / 50.0

        # CuPy
        t_cp = None
        if CUPY_AVAILABLE:
            a_cp = cp.array(a_host)
            b_cp = cp.array(b_host)
            # Warmup
            for _ in range(5):
                res_cp = a_cp + b_cp
                res_cp = cp.exp(res_cp)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                res_cp = a_cp + b_cp
                res_cp = cp.exp(res_cp)
            cp.cuda.Stream.null.synchronize()
            t_cp = (time.perf_counter() - t0) / 50.0

        # Output times
        print(f"  NumPy (CPU) time:        {t_np * 1000:.3f} ms")
        print(f"  Hephaestus (WGPU) time:  {t_ph * 1000:.3f} ms (Speedup vs NumPy: {t_np / t_ph:.2f}x)")
        if CUPY_AVAILABLE:
            print(f"  CuPy (CUDA GPU) time:    {t_cp * 1000:.3f} ms (Speedup vs NumPy: {t_np / t_cp:.2f}x)")

if __name__ == "__main__":
    run_bench()
