import time
import os
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
    print("       Hephaestus (WGPU/CUDA) vs NumPy (CPU) vs CuPy Benchmark        ")
    print("======================================================================")
    print(f"CuPy available: {CUPY_AVAILABLE}\n")

    devices = []
    requested = os.environ.get("HEPHAESTUS_BACKENDS", "wgpu,cuda")
    for backend in [name.strip() for name in requested.split(",") if name.strip()]:
        try:
            dev = ph.Device(backend)
            devices.append((backend, dev))
            print(f"Initialized Hephaestus Device (Backend: {dev.backend_name})")
        except RuntimeError as exc:
            print(f"Skipping Hephaestus backend {backend}: {exc}")

    if not devices:
        raise RuntimeError("no Hephaestus backend initialized")

    sizes = [1_000, 10_000, 100_000, 1_000_000, 5_000_000]

    for size in sizes:
        print(f"\n--- Array Size: {size:,} f32 elements ---")

        # 1. Correctness Validation
        # Generate random host data
        a_host = np.random.uniform(-1.0, 1.0, size).astype(np.float32)
        b_host = np.random.uniform(-1.0, 1.0, size).astype(np.float32)

        add_np = a_host + b_host
        exp_np = np.exp(a_host)
        scalar_add_np = a_host + 5.0
        sum_np = np.sum(a_host)

        hephaestus_times = []
        for backend, dev in devices:
            a_ph = ph.Array(a_host.tolist(), dev)
            b_ph = ph.Array(b_host.tolist(), dev)

            add_ph = a_ph + b_ph
            np.testing.assert_allclose(np.array(add_ph.tolist(), dtype=np.float32), add_np, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(np.array(a_ph.exp().tolist(), dtype=np.float32), exp_np, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(np.array((a_ph + 5.0).tolist(), dtype=np.float32), scalar_add_np, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(a_ph.sum().tolist()[0], sum_np, rtol=1e-3, atol=1e-3)

            for _ in range(5):
                res_ph = a_ph + b_ph
                res_ph = res_ph.exp()
            t0 = time.perf_counter()
            for _ in range(50):
                res_ph = a_ph + b_ph
                res_ph = res_ph.exp()
            res_ph.tolist()
            hephaestus_times.append((backend, (time.perf_counter() - t0) / 50.0))

        print("-> Correctness validation PASSED!")

        # 2. Performance benchmarking
        # NumPy
        t0 = time.perf_counter()
        for _ in range(50):
            res_np = a_host + b_host
            res_np = np.exp(res_np)
        t_np = (time.perf_counter() - t0) / 50.0

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
        for backend, elapsed in hephaestus_times:
            print(f"  Hephaestus ({backend}) time: {elapsed * 1000:.3f} ms (Speedup vs NumPy: {t_np / elapsed:.2f}x)")
        if CUPY_AVAILABLE:
            print(f"  CuPy (CUDA GPU) time:    {t_cp * 1000:.3f} ms (Speedup vs NumPy: {t_np / t_cp:.2f}x)")

if __name__ == "__main__":
    run_bench()
