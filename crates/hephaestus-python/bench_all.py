import time
import numpy as np
import pyhephaestus as ph
import leto_python as lp

# Try importing cupy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

def run_bench():
    print("======================================================================")
    print("            Atlas Array Substrates Performance Comparison             ")
    print("======================================================================")
    print(f"CuPy available: {CUPY_AVAILABLE}")
    
    # Initialize WGPU device
    dev = ph.Device()
    print(f"Initialized Hephaestus Device (Backend: {dev.backend_name})\n")

    # Benchmarking parameters
    # We use a 2D shape for compatibility with Leto's 2D-only Python bindings
    sizes_2d = [
        (100, 100),    # 10,000 elements
        (500, 500),    # 250,000 elements
        (1000, 1000),  # 1,000,000 elements
        (2000, 2500)   # 5,000,000 elements
    ]

    results = []

    for rows, cols in sizes_2d:
        size = rows * cols
        print(f"\n--- Benchmark size: {rows}x{cols} ({size:,} f32 elements) ---")

        # ─── 1. Prepare Inputs ───
        # Deterministic inputs on host
        a_np = (np.arange(size, dtype=np.float32) * 0.0001).reshape(rows, cols)
        b_np = (np.arange(size, dtype=np.float32) * 0.0002).reshape(rows, cols)

        # Upload to Hephaestus
        a_ph = ph.Array(a_np.ravel().tolist(), dev)
        b_ph = ph.Array(b_np.ravel().tolist(), dev)

        # Upload to CuPy
        if CUPY_AVAILABLE:
            a_cp = cp.array(a_np)
            b_cp = cp.array(b_np)

        # ─── 2. Correctness Checks ───
        # Addition
        add_np = a_np + b_np
        add_lp = lp.add(a_np, b_np)
        add_ph = a_ph + b_ph
        np.testing.assert_allclose(add_lp, add_np, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.array(add_ph.tolist(), dtype=np.float32), add_np.ravel(), rtol=1e-5, atol=1e-5)

        # Division
        div_np = (a_np + 1.0) / (b_np + 1.0)
        div_lp = lp.div(a_np + 1.0, b_np + 1.0)
        a_ph_p1 = a_ph + 1.0
        b_ph_p1 = b_ph + 1.0
        div_ph = a_ph_p1 / b_ph_p1
        np.testing.assert_allclose(div_lp, div_np, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.array(div_ph.tolist(), dtype=np.float32), div_np.ravel(), rtol=1e-5, atol=1e-5)

        # Power
        pow_np = (a_np + 1.0) ** 2.0
        pow_ph = (a_ph + 1.0) ** 2.0
        np.testing.assert_allclose(np.array(pow_ph.tolist(), dtype=np.float32), pow_np.ravel(), rtol=1e-5, atol=1e-5)

        # Unary Exp
        exp_np = np.exp(a_np)
        exp_ph = a_ph.exp()
        np.testing.assert_allclose(np.array(exp_ph.tolist(), dtype=np.float32), exp_np.ravel(), rtol=1e-5, atol=1e-5)

        # Reduction Sum
        sum_np = np.sum(a_np)
        sum_lp = lp.sum(a_np)
        sum_ph = a_ph.sum()
        np.testing.assert_allclose(sum_lp, sum_np, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(sum_ph.tolist()[0], sum_np, rtol=1e-3, atol=1e-3)

        # Reduction Mean
        mean_np = np.mean(a_np)
        mean_ph = a_ph.mean()
        np.testing.assert_allclose(mean_ph.tolist()[0], mean_np, rtol=1e-3, atol=1e-3)

        # Matrix Multiplication (matmul)
        b_np_matmul = (np.arange(size, dtype=np.float32) * 0.0002).reshape(cols, rows)
        matmul_np = np.matmul(a_np, b_np_matmul)
        matmul_lp = lp.matmul(a_np, b_np_matmul)
        a_ph_matmul = a_ph.reshape([rows, cols])
        b_ph_matmul = ph.Array(b_np_matmul.ravel().tolist(), dev).reshape([cols, rows])
        matmul_ph = a_ph_matmul.matmul(b_ph_matmul)
        np.testing.assert_allclose(matmul_lp, matmul_np, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.array(matmul_ph.tolist(), dtype=np.float32), matmul_np.ravel(), rtol=1e-3, atol=1e-3)

        # Dot Product
        dot_np = np.dot(a_np.ravel(), b_np.ravel())
        dot_lp = lp.dot(a_np.ravel(), b_np.ravel())
        dot_ph = a_ph.dot(b_ph)
        np.testing.assert_allclose(dot_lp, dot_np, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(dot_ph.tolist()[0], dot_np, rtol=1e-3, atol=1e-3)

        # Trace (requires square matrix)
        a_np_sq = (np.arange(rows * rows, dtype=np.float32) * 0.0001).reshape(rows, rows)
        trace_np = np.trace(a_np_sq)
        trace_lp = lp.trace(a_np_sq.astype(np.float64))
        a_ph_sq = ph.Array(a_np_sq.ravel().tolist(), dev).reshape([rows, rows])
        trace_ph = a_ph_sq.trace()
        np.testing.assert_allclose(trace_lp, trace_np, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(trace_ph.tolist()[0], trace_np, rtol=1e-3, atol=1e-3)

        # Norms
        norm_l1_np = np.sum(np.abs(a_np_sq))
        norm_l1_lp = lp.norm(a_np_sq.astype(np.float64), ord="1")
        norm_l1_ph = a_ph_sq.norm_l1()
        np.testing.assert_allclose(norm_l1_lp, norm_l1_np, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(norm_l1_ph.tolist()[0], norm_l1_np, rtol=1e-3, atol=1e-3)

        norm_l2_np = np.linalg.norm(a_np_sq, ord="fro")
        norm_l2_lp = lp.norm(a_np_sq.astype(np.float64), ord="fro")
        norm_l2_ph = a_ph_sq.norm_l2()
        np.testing.assert_allclose(norm_l2_lp, norm_l2_np, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(norm_l2_ph.tolist()[0], norm_l2_np, rtol=1e-3, atol=1e-3)

        norm_max_np = np.max(np.abs(a_np_sq))
        norm_max_lp = lp.norm(a_np_sq.astype(np.float64), ord="max")
        norm_max_ph = a_ph_sq.norm_max()
        np.testing.assert_allclose(norm_max_lp, norm_max_np, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(norm_max_ph.tolist()[0], norm_max_np, rtol=1e-3, atol=1e-3)

        print("-> Correctness validation checks PASSED for all substrates!")

        # ─── 3. Timing Benchmarks ───
        # Number of iterations
        iters = 50

        # --- Elementwise Add + Exp ---
        # NumPy
        t0 = time.perf_counter()
        for _ in range(iters):
            out_np = a_np + b_np
            out_np = np.exp(out_np)
        t_np_add_exp = (time.perf_counter() - t0) / iters

        # Leto (Add only)
        t0 = time.perf_counter()
        for _ in range(iters):
            out_lp = lp.add(a_np, b_np)
        t_lp_add = (time.perf_counter() - t0) / iters

        # Hephaestus (WGPU)
        for _ in range(5):  # Warmup
            out_ph = a_ph + b_ph
            out_ph = out_ph.exp()
        t0 = time.perf_counter()
        for _ in range(iters):
            out_ph = a_ph + b_ph
            out_ph = out_ph.exp()
        out_ph.tolist()  # Synchronize
        t_ph_add_exp = (time.perf_counter() - t0) / iters

        # CuPy
        t_cp_add_exp = None
        if CUPY_AVAILABLE:
            for _ in range(5):
                out_cp = a_cp + b_cp
                out_cp = cp.exp(out_cp)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                out_cp = a_cp + b_cp
                out_cp = cp.exp(out_cp)
            cp.cuda.Stream.null.synchronize()
            t_cp_add_exp = (time.perf_counter() - t0) / iters

        # --- Elementwise Div + Pow ---
        # NumPy
        t0 = time.perf_counter()
        for _ in range(iters):
            out_np = (a_np + 1.0) / (b_np + 1.0)
            out_np = out_np ** 2.0
        t_np_div_pow = (time.perf_counter() - t0) / iters

        # Leto (Div only)
        t0 = time.perf_counter()
        for _ in range(iters):
            out_lp = lp.div(a_np + 1.0, b_np + 1.0)
        t_lp_div = (time.perf_counter() - t0) / iters

        # Hephaestus (WGPU)
        for _ in range(5):
            out_ph = (a_ph + 1.0) / (b_ph + 1.0)
            out_ph = out_ph ** 2.0
        t0 = time.perf_counter()
        for _ in range(iters):
            out_ph = (a_ph + 1.0) / (b_ph + 1.0)
            out_ph = out_ph ** 2.0
        out_ph.tolist()
        t_ph_div_pow = (time.perf_counter() - t0) / iters

        # CuPy
        t_cp_div_pow = None
        if CUPY_AVAILABLE:
            for _ in range(5):
                out_cp = (a_cp + 1.0) / (b_cp + 1.0)
                out_cp = out_cp ** 2.0
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                out_cp = (a_cp + 1.0) / (b_cp + 1.0)
                out_cp = out_cp ** 2.0
            cp.cuda.Stream.null.synchronize()
            t_cp_div_pow = (time.perf_counter() - t0) / iters

        # --- Reduction (Sum + Mean) ---
        # NumPy
        t0 = time.perf_counter()
        for _ in range(iters):
            out_sum = np.sum(a_np)
            out_mean = np.mean(a_np)
        t_np_red = (time.perf_counter() - t0) / iters

        # Leto (Sum only)
        t0 = time.perf_counter()
        for _ in range(iters):
            out_sum = lp.sum(a_np)
        t_lp_red = (time.perf_counter() - t0) / iters

        # Hephaestus (WGPU)
        for _ in range(5):
            out_sum = a_ph.sum()
            out_mean = a_ph.mean()
        t0 = time.perf_counter()
        for _ in range(iters):
            out_sum = a_ph.sum()
            out_mean = a_ph.mean()
        out_mean.tolist()
        t_ph_red = (time.perf_counter() - t0) / iters

        # CuPy
        t_cp_red = None
        if CUPY_AVAILABLE:
            for _ in range(5):
                out_sum = cp.sum(a_cp)
                out_mean = cp.mean(a_cp)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                out_sum = cp.sum(a_cp)
                out_mean = cp.mean(a_cp)
            cp.cuda.Stream.null.synchronize()
            t_cp_red = (time.perf_counter() - t0) / iters

        # --- Matrix Multiplication (matmul) ---
        iters_matmul = 5 if rows >= 1000 else 20
        # NumPy
        t0 = time.perf_counter()
        for _ in range(iters_matmul):
            out_np = np.matmul(a_np, b_np_matmul)
        t_np_matmul = (time.perf_counter() - t0) / iters_matmul

        # Leto
        t0 = time.perf_counter()
        for _ in range(iters_matmul):
            out_lp = lp.matmul(a_np, b_np_matmul)
        t_lp_matmul = (time.perf_counter() - t0) / iters_matmul

        # Hephaestus
        for _ in range(3):
            out_ph = a_ph_matmul.matmul(b_ph_matmul)
        t0 = time.perf_counter()
        for _ in range(iters_matmul):
            out_ph = a_ph_matmul.matmul(b_ph_matmul)
        out_ph.tolist()
        t_ph_matmul = (time.perf_counter() - t0) / iters_matmul

        # CuPy
        t_cp_matmul = None
        if CUPY_AVAILABLE:
            a_cp_matmul = a_cp.reshape(rows, cols)
            b_cp_matmul = cp.array(b_np_matmul)
            for _ in range(3):
                out_cp = cp.matmul(a_cp_matmul, b_cp_matmul)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters_matmul):
                out_cp = cp.matmul(a_cp_matmul, b_cp_matmul)
            cp.cuda.Stream.null.synchronize()
            t_cp_matmul = (time.perf_counter() - t0) / iters_matmul

        # --- Vector Dot Product ---
        # NumPy
        t0 = time.perf_counter()
        for _ in range(iters):
            out_np = np.dot(a_np.ravel(), b_np.ravel())
        t_np_dot = (time.perf_counter() - t0) / iters

        # Leto
        t0 = time.perf_counter()
        for _ in range(iters):
            out_lp = lp.dot(a_np.ravel(), b_np.ravel())
        t_lp_dot = (time.perf_counter() - t0) / iters

        # Hephaestus
        for _ in range(5):
            out_ph = a_ph.dot(b_ph)
        t0 = time.perf_counter()
        for _ in range(iters):
            out_ph = a_ph.dot(b_ph)
        out_ph.tolist()
        t_ph_dot = (time.perf_counter() - t0) / iters

        # CuPy
        t_cp_dot = None
        if CUPY_AVAILABLE:
            for _ in range(5):
                out_cp = cp.dot(a_cp.ravel(), b_cp.ravel())
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                out_cp = cp.dot(a_cp.ravel(), b_cp.ravel())
            cp.cuda.Stream.null.synchronize()
            t_cp_dot = (time.perf_counter() - t0) / iters

        # --- Matrix Trace ---
        # NumPy
        t0 = time.perf_counter()
        for _ in range(iters):
            out_np = np.trace(a_np_sq)
        t_np_trace = (time.perf_counter() - t0) / iters

        # Leto
        t0 = time.perf_counter()
        for _ in range(iters):
            out_lp = lp.trace(a_np_sq.astype(np.float64))
        t_lp_trace = (time.perf_counter() - t0) / iters

        # Hephaestus
        for _ in range(5):
            out_ph = a_ph_sq.trace()
        t0 = time.perf_counter()
        for _ in range(iters):
            out_ph = a_ph_sq.trace()
        out_ph.tolist()
        t_ph_trace = (time.perf_counter() - t0) / iters

        # CuPy
        t_cp_trace = None
        if CUPY_AVAILABLE:
            a_cp_sq = cp.array(a_np_sq)
            for _ in range(5):
                out_cp = cp.trace(a_cp_sq)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                out_cp = cp.trace(a_cp_sq)
            cp.cuda.Stream.null.synchronize()
            t_cp_trace = (time.perf_counter() - t0) / iters

        # --- Matrix Frobenius Norm (L2) ---
        # NumPy
        t0 = time.perf_counter()
        for _ in range(iters):
            out_np = np.linalg.norm(a_np_sq, ord="fro")
        t_np_norm_l2 = (time.perf_counter() - t0) / iters

        # Leto
        t0 = time.perf_counter()
        for _ in range(iters):
            out_lp = lp.norm(a_np_sq.astype(np.float64), ord="fro")
        t_lp_norm_l2 = (time.perf_counter() - t0) / iters

        # Hephaestus
        for _ in range(5):
            out_ph = a_ph_sq.norm_l2()
        t0 = time.perf_counter()
        for _ in range(iters):
            out_ph = a_ph_sq.norm_l2()
        out_ph.tolist()
        t_ph_norm_l2 = (time.perf_counter() - t0) / iters

        # CuPy
        t_cp_norm_l2 = None
        if CUPY_AVAILABLE:
            for _ in range(5):
                out_cp = cp.linalg.norm(a_cp_sq)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                out_cp = cp.linalg.norm(a_cp_sq)
            cp.cuda.Stream.null.synchronize()
            t_cp_norm_l2 = (time.perf_counter() - t0) / iters

        # Store measurements (in milliseconds)
        results.append({
            'size': size,
            'rows': rows,
            'cols': cols,
            'add_exp': {
                'np': t_np_add_exp * 1000,
                'lp_add': t_lp_add * 1000,
                'ph': t_ph_add_exp * 1000,
                'cp': t_cp_add_exp * 1000 if t_cp_add_exp else None
            },
            'div_pow': {
                'np': t_np_div_pow * 1000,
                'lp_div': t_lp_div * 1000,
                'ph': t_ph_div_pow * 1000,
                'cp': t_cp_div_pow * 1000 if t_cp_div_pow else None
            },
            'reduction': {
                'np': t_np_red * 1000,
                'lp_sum': t_lp_red * 1000,
                'ph': t_ph_red * 1000,
                'cp': t_cp_red * 1000 if t_cp_red else None
            },
            'matmul': {
                'np': t_np_matmul * 1000,
                'lp': t_lp_matmul * 1000,
                'ph': t_ph_matmul * 1000,
                'cp': t_cp_matmul * 1000 if t_cp_matmul else None
            },
            'dot': {
                'np': t_np_dot * 1000,
                'lp': t_lp_dot * 1000,
                'ph': t_ph_dot * 1000,
                'cp': t_cp_dot * 1000 if t_cp_dot else None
            },
            'trace': {
                'np': t_np_trace * 1000,
                'lp': t_lp_trace * 1000,
                'ph': t_ph_trace * 1000,
                'cp': t_cp_trace * 1000 if t_cp_trace else None
            },
            'norm_l2': {
                'np': t_np_norm_l2 * 1000,
                'lp': t_lp_norm_l2 * 1000,
                'ph': t_ph_norm_l2 * 1000,
                'cp': t_cp_norm_l2 * 1000 if t_cp_norm_l2 else None
            }
        })

    # Print a markdown formatted comparison table
    print("\n\n======================================================================")
    print("                      Benchmark Comparison Table                      ")
    print("======================================================================")
    
    print("\n### 1. Elementwise Add + Exp (ms)")
    print("| Array Size | NumPy (CPU) | Leto CPU (Add-only) | Hephaestus (WGPU GPU) | CuPy (CUDA) | Speedup vs Leto |")
    print("|---|---|---|---|---|---|")
    for r in results:
        lp_val = f"{r['add_exp']['lp_add']:.3f}"
        ph_val = f"{r['add_exp']['ph']:.3f}"
        cp_val = f"{r['add_exp']['cp']:.3f}" if r['add_exp']['cp'] else "N/A"
        speedup = f"{r['add_exp']['lp_add'] / r['add_exp']['ph']:.2f}x"
        print(f"| {r['size']:,} | {r['add_exp']['np']:.3f} | {lp_val} | {ph_val} | {cp_val} | {speedup} |")

    print("\n### 2. Elementwise Div + Pow (ms)")
    print("| Array Size | NumPy (CPU) | Leto CPU (Div-only) | Hephaestus (WGPU GPU) | CuPy (CUDA) | Speedup vs Leto |")
    print("|---|---|---|---|---|---|")
    for r in results:
        lp_val = f"{r['div_pow']['lp_div']:.3f}"
        ph_val = f"{r['div_pow']['ph']:.3f}"
        cp_val = f"{r['div_pow']['cp']:.3f}" if r['div_pow']['cp'] else "N/A"
        speedup = f"{r['div_pow']['lp_div'] / r['div_pow']['ph']:.2f}x"
        print(f"| {r['size']:,} | {r['div_pow']['np']:.3f} | {lp_val} | {ph_val} | {cp_val} | {speedup} |")

    print("\n### 3. Reductions (ms)")
    print("| Array Size | NumPy (CPU) | Leto CPU (Sum-only) | Hephaestus (WGPU GPU) | CuPy (CUDA) | Speedup vs Leto |")
    print("|---|---|---|---|---|---|")
    for r in results:
        lp_val = f"{r['reduction']['lp_sum']:.3f}"
        ph_val = f"{r['reduction']['ph']:.3f}"
        cp_val = f"{r['reduction']['cp']:.3f}" if r['reduction']['cp'] else "N/A"
        speedup = f"{r['reduction']['lp_sum'] / r['reduction']['ph']:.2f}x"
        print(f"| {r['size']:,} | {r['reduction']['np']:.3f} | {lp_val} | {ph_val} | {cp_val} | {speedup} |")

    print("\n### 4. Matrix Multiplication (ms)")
    print("| Shape | NumPy (CPU) | Leto CPU | Hephaestus (WGPU GPU) | CuPy (CUDA) | Speedup vs Leto |")
    print("|---|---|---|---|---|---|")
    for r in results:
        lp_val = f"{r['matmul']['lp']:.3f}"
        ph_val = f"{r['matmul']['ph']:.3f}"
        cp_val = f"{r['matmul']['cp']:.3f}" if r['matmul']['cp'] else "N/A"
        speedup = f"{r['matmul']['lp'] / r['matmul']['ph']:.2f}x"
        print(f"| ({r['rows']}x{r['cols']}) x ({r['cols']}x{r['rows']}) | {r['matmul']['np']:.3f} | {lp_val} | {ph_val} | {cp_val} | {speedup} |")

    print("\n### 5. Vector Dot Product (ms)")
    print("| Vector Size | NumPy (CPU) | Leto CPU | Hephaestus (WGPU GPU) | CuPy (CUDA) | Speedup vs Leto |")
    print("|---|---|---|---|---|---|")
    for r in results:
        lp_val = f"{r['dot']['lp']:.3f}"
        ph_val = f"{r['dot']['ph']:.3f}"
        cp_val = f"{r['dot']['cp']:.3f}" if r['dot']['cp'] else "N/A"
        speedup = f"{r['dot']['lp'] / r['dot']['ph']:.2f}x"
        print(f"| {r['size']:,} | {r['dot']['np']:.3f} | {lp_val} | {ph_val} | {cp_val} | {speedup} |")

    print("\n### 6. Matrix Trace (ms)")
    print("| Matrix Shape | NumPy (CPU) | Leto CPU | Hephaestus (WGPU GPU) | CuPy (CUDA) | Speedup vs Leto |")
    print("|---|---|---|---|---|---|")
    for r in results:
        lp_val = f"{r['trace']['lp']:.3f}"
        ph_val = f"{r['trace']['ph']:.3f}"
        cp_val = f"{r['trace']['cp']:.3f}" if r['trace']['cp'] else "N/A"
        speedup = f"{r['trace']['lp'] / r['trace']['ph']:.2f}x"
        print(f"| {r['rows']}x{r['rows']} | {r['trace']['np']:.3f} | {lp_val} | {ph_val} | {cp_val} | {speedup} |")

    print("\n### 7. Matrix Frobenius Norm (L2) (ms)")
    print("| Matrix Shape | NumPy (CPU) | Leto CPU | Hephaestus (WGPU GPU) | CuPy (CUDA) | Speedup vs Leto |")
    print("|---|---|---|---|---|---|")
    for r in results:
        lp_val = f"{r['norm_l2']['lp']:.3f}"
        ph_val = f"{r['norm_l2']['ph']:.3f}"
        cp_val = f"{r['norm_l2']['cp']:.3f}" if r['norm_l2']['cp'] else "N/A"
        speedup = f"{r['norm_l2']['lp'] / r['norm_l2']['ph']:.2f}x"
        print(f"| {r['rows']}x{r['rows']} | {r['norm_l2']['np']:.3f} | {lp_val} | {ph_val} | {cp_val} | {speedup} |")

if __name__ == "__main__":
    run_bench()
