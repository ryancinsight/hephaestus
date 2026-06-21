//! Comparative benchmark of Hephaestus GPU (CUDA) vs CPU (Leto & ndarray & nalgebra).
//!
//! Validates results across all backends to ensure correctness and outputs timing.

use std::hint::black_box;
use std::time::{Duration, Instant};

use hephaestus_core::BlockWidth;
use hephaestus_cuda::{
    binary_elementwise_into, cholesky_decompose_blocked, cumsum_into, det, dot, kron_into,
    lu_decompose, matmul_into, matpow, matrix_rank, max_axis_into, mean_axis_into, min_axis_into,
    norm_l1, norm_l2, norm_max, qr_decompose, qr_decompose_blocked, reduction, sum_axis_into,
    symmetric_eigen_jacobi, trace, unary_elementwise_into, AddOp, ComputeDevice, CudaDevice, ExpOp,
    StridedOperand as CudaStridedOperand, SumOp,
};
use nalgebra::DMatrix;
use ndarray::Array2 as NdArray2;
use ndarray::{Array1 as NdArray1, Axis};

const LEN: usize = 1 << 20; // 1,048,576 elements for elementwise
const LINALG_LEN: usize = 1 << 16; // 65,536 elements for dot/norms
const ITERS: usize = 50;

fn wait_cuda(_device: &CudaDevice) {
    // CUDA launches on the default stream are synchronized at download/transfer.
}

fn elapsed_per_iter(elapsed: Duration) -> Duration {
    elapsed / u32::try_from(ITERS).expect("invariant: iteration count fits u32")
}

fn assert_close_slice(got: &[f32], expected: &[f32], abs_tol: f32, rel_tol: f32) {
    assert_eq!(got.len(), expected.len());
    for (index, (&got, &expected)) in got.iter().zip(expected.iter()).enumerate() {
        let tolerance = abs_tol.max(rel_tol * expected.abs().max(1.0));
        assert!(
            (got - expected).abs() <= tolerance,
            "slice mismatch at {index}: got {got}, expected {expected}, tolerance {tolerance}"
        );
    }
}

// ─── CPU References (ndarray / nalgebra) ───

fn nalgebra_kron(a: &DMatrix<f32>, b: &DMatrix<f32>) -> DMatrix<f32> {
    let rows = a.nrows() * b.nrows();
    let cols = a.ncols() * b.ncols();
    let mut out = DMatrix::zeros(rows, cols);
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let scale = a[(i, j)];
            let row_base = i * b.nrows();
            let col_base = j * b.ncols();
            for k in 0..b.nrows() {
                for l in 0..b.ncols() {
                    out[(row_base + k, col_base + l)] = scale * b[(k, l)];
                }
            }
        }
    }
    out
}

fn ndarray_matpow(matrix: &NdArray2<f32>, exponent: u32) -> NdArray2<f32> {
    let n = matrix.nrows();
    let mut result = NdArray2::zeros((n, n));
    for i in 0..n {
        result[(i, i)] = 1.0;
    }
    if exponent == 0 {
        return result;
    }

    let mut base = matrix.clone();
    let mut remaining = exponent;
    loop {
        if remaining & 1 == 1 {
            result = result.dot(&base);
        }
        remaining >>= 1;
        if remaining == 0 {
            break;
        }
        base = base.dot(&base);
    }
    result
}

fn nalgebra_sum_axis(matrix: &DMatrix<f32>, axis: usize) -> Vec<f32> {
    if axis == 0 {
        let mut out = vec![0.0f32; matrix.ncols()];
        for col in 0..matrix.ncols() {
            let mut sum = 0.0f32;
            for row in 0..matrix.nrows() {
                sum += matrix[(row, col)];
            }
            out[col] = sum;
        }
        out
    } else {
        let mut out = vec![0.0f32; matrix.nrows()];
        for row in 0..matrix.nrows() {
            let mut sum = 0.0f32;
            for col in 0..matrix.ncols() {
                sum += matrix[(row, col)];
            }
            out[row] = sum;
        }
        out
    }
}

fn main() {
    println!("=== Starting comparative benchmarks (CUDA GPU vs CPU) ===");
    println!("Iterations: {ITERS}");

    // Setup CUDA
    let cuda_dev = match CudaDevice::try_default() {
        Ok(d) => {
            println!("CUDA GPU Backend acquired successfully.\n");
            d
        }
        Err(e) => {
            println!("CUDA GPU Backend not available: {e}\n");
            return;
        }
    };

    // ────────────────────────────────────────────────────────────────────────
    // 1. Elementwise Add (f32, size 1<<20)
    // ────────────────────────────────────────────────────────────────────────
    {
        println!("--- Benchmarking: Elementwise Add (f32, N={LEN}) ---");
        let host_a: Vec<f32> = (0..LEN).map(|i| (i as f32 * 0.731 + 1.0) * 1e-7).collect();
        let host_b: Vec<f32> = (0..LEN).map(|i| (i as f32 * 0.317 + 2.0) * 1e-7).collect();

        // Leto (CPU)
        let leto_a = leto::Array::from_shape_vec([LEN], host_a.clone()).unwrap();
        let leto_b = leto::Array::from_shape_vec([LEN], host_b.clone()).unwrap();
        let mut leto_out = leto::Array::zeros([LEN]);
        leto_ops::add(&leto_a.view(), &leto_b.view(), &mut leto_out.view_mut()).unwrap();
        let nd_a = NdArray1::from_vec(host_a.clone());
        let nd_b = NdArray1::from_vec(host_b.clone());

        // CUDA (GPU)
        let cu_a = cuda_dev.upload(&host_a).unwrap();
        let cu_b = cuda_dev.upload(&host_b).unwrap();
        let cu_out = cuda_dev.alloc_zeroed::<f32>(LEN).unwrap();
        binary_elementwise_into::<AddOp, f32>(
            &cuda_dev,
            &cu_a,
            &cu_b,
            &cu_out,
            BlockWidth::DEFAULT,
        )
        .unwrap();
        wait_cuda(&cuda_dev);
        let mut got_cuda = vec![0.0f32; LEN];
        cuda_dev.download(&cu_out, &mut got_cuda).unwrap();
        assert_close_slice(
            &got_cuda,
            leto::Storage::as_slice(leto_out.storage()),
            1e-5,
            1e-5,
        );

        // Benchmark CPU Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let mut leto_out = leto::Array::zeros([LEN]);
            leto_ops::add(
                black_box(&leto_a.view()),
                black_box(&leto_b.view()),
                black_box(&mut leto_out.view_mut()),
            )
            .unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark CPU ndarray
        let t_ndarray = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(&nd_a) + black_box(&nd_b);
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_ndarray.elapsed()).as_nanos()
        );

        // Benchmark CUDA GPU
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            binary_elementwise_into::<AddOp, f32>(
                &cuda_dev,
                black_box(&cu_a),
                black_box(&cu_b),
                black_box(&cu_out),
                BlockWidth::DEFAULT,
            )
            .unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 2. Elementwise Exp (f32, size 1<<20)
    // ────────────────────────────────────────────────────────────────────────
    {
        println!("--- Benchmarking: Elementwise Exp (f32, N={LEN}) ---");
        let host_a: Vec<f32> = (0..LEN).map(|i| (i as f32 * 0.731 + 1.0) * 1e-7).collect();

        // Leto (CPU)
        let leto_a = leto::Array::from_shape_vec([LEN], host_a.clone()).unwrap();
        let mut leto_out = leto::Array::zeros([LEN]);
        leto_ops::unary_map_into(leto_ops::ExpOp, &leto_a.view(), &mut leto_out.view_mut())
            .unwrap();
        let nd_a = NdArray1::from_vec(host_a.clone());

        // CUDA (GPU)
        let cu_a = cuda_dev.upload(&host_a).unwrap();
        let cu_out = cuda_dev.alloc_zeroed::<f32>(LEN).unwrap();
        unary_elementwise_into::<ExpOp, f32>(&cuda_dev, &cu_a, &cu_out, BlockWidth::DEFAULT)
            .unwrap();
        wait_cuda(&cuda_dev);
        let mut got_cuda = vec![0.0f32; LEN];
        cuda_dev.download(&cu_out, &mut got_cuda).unwrap();
        assert_close_slice(
            &got_cuda,
            leto::Storage::as_slice(leto_out.storage()),
            1e-5,
            1e-5,
        );

        // Benchmark CPU Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let mut leto_out = leto::Array::zeros([LEN]);
            leto_ops::unary_map_into(
                leto_ops::ExpOp,
                black_box(&leto_a.view()),
                black_box(&mut leto_out.view_mut()),
            )
            .unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark CPU ndarray
        let t_ndarray = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(&nd_a).mapv(f32::exp);
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_ndarray.elapsed()).as_nanos()
        );

        // Benchmark CUDA GPU
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            unary_elementwise_into::<ExpOp, f32>(
                &cuda_dev,
                black_box(&cu_a),
                black_box(&cu_out),
                BlockWidth::DEFAULT,
            )
            .unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 3. Sum Reduction (f32, size 1<<20)
    // ────────────────────────────────────────────────────────────────────────
    {
        println!("--- Benchmarking: Sum Reduction (f32, N={LEN}) ---");
        let host_a: Vec<f32> = (0..LEN).map(|i| (i as f32 * 0.731 + 1.0) * 1e-7).collect();

        // Leto (CPU)
        let leto_a = leto::Array::from_shape_vec([LEN], host_a.clone()).unwrap();
        let leto_sum = leto_ops::sum(&leto_a.view());
        let nd_a = NdArray1::from_vec(host_a.clone());

        // CUDA (GPU)
        let cu_a = cuda_dev.upload(&host_a).unwrap();
        let cu_out = reduction::<SumOp, f32>(&cuda_dev, &cu_a).unwrap();
        wait_cuda(&cuda_dev);
        let mut got_cuda = vec![0.0f32; 1];
        cuda_dev.download(&cu_out, &mut got_cuda).unwrap();
        let diff = (got_cuda[0] - leto_sum).abs();
        assert!(
            diff <= 1e-2,
            "Sum mismatch: got_cuda {}, expected {}",
            got_cuda[0],
            leto_sum
        );

        // Benchmark CPU Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _val = leto_ops::sum(black_box(&leto_a.view()));
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark CPU ndarray
        let t_ndarray = Instant::now();
        for _ in 0..ITERS {
            let _val = black_box(&nd_a).sum();
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_ndarray.elapsed()).as_nanos()
        );

        // Benchmark CUDA GPU
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            let _res = reduction::<SumOp, f32>(&cuda_dev, black_box(&cu_a)).unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 4. Axis Reductions (f32, 256x256 shape, axis 0)
    // ────────────────────────────────────────────────────────────────────────
    let axis_n = 256;
    let host_axis: Vec<f32> = (0..(axis_n * axis_n))
        .map(|i| (i as f32 * 0.731 + 1.0) * 1e-4)
        .collect();
    let leto_axis = leto::Array::from_shape_vec([axis_n, axis_n], host_axis.clone()).unwrap();
    let nd_axis = NdArray2::from_shape_vec((axis_n, axis_n), host_axis.clone()).unwrap();
    let na_axis = DMatrix::from_vec(axis_n, axis_n, host_axis.clone());

    for op_idx in 0..4 {
        let op_name = match op_idx {
            0 => "Sum",
            1 => "Min",
            2 => "Max",
            _ => "Mean",
        };
        println!("--- Benchmarking: Axis {op_name} (f32, {axis_n}x{axis_n} over axis 0) ---");

        // Leto (CPU)
        let mut leto_out = leto::Array::zeros([1, axis_n]);
        match op_idx {
            0 => leto_ops::sum_axis_into(&leto_axis.view(), 0, &mut leto_out.view_mut()).unwrap(),
            1 => leto_ops::min_axis_into(&leto_axis.view(), 0, &mut leto_out.view_mut()).unwrap(),
            2 => leto_ops::max_axis_into(&leto_axis.view(), 0, &mut leto_out.view_mut()).unwrap(),
            _ => leto_ops::mean_axis_into(&leto_axis.view(), 0, &mut leto_out.view_mut()).unwrap(),
        }

        // CUDA (GPU)
        let cu_axis = cuda_dev.upload(&host_axis).unwrap();
        let cu_out = cuda_dev.alloc_zeroed::<f32>(axis_n).unwrap();
        let axis_input = CudaStridedOperand {
            buffer: &cu_axis,
            layout: &leto::Layout::c_contiguous([axis_n, axis_n]).unwrap(),
        };
        let axis_output_layout = leto::Layout::c_contiguous([1, axis_n]).unwrap();
        let axis_output = CudaStridedOperand {
            buffer: &cu_out,
            layout: &axis_output_layout,
        };
        match op_idx {
            0 => sum_axis_into(&cuda_dev, axis_input, 0, axis_output, BlockWidth::DEFAULT).unwrap(),
            1 => min_axis_into(&cuda_dev, axis_input, 0, axis_output, BlockWidth::DEFAULT).unwrap(),
            2 => max_axis_into(&cuda_dev, axis_input, 0, axis_output, BlockWidth::DEFAULT).unwrap(),
            _ => {
                mean_axis_into(&cuda_dev, axis_input, 0, axis_output, BlockWidth::DEFAULT).unwrap()
            }
        }
        wait_cuda(&cuda_dev);
        let mut got_cuda = vec![0.0f32; axis_n];
        cuda_dev.download(&cu_out, &mut got_cuda).unwrap();
        assert_close_slice(
            &got_cuda,
            leto::Storage::as_slice(leto_out.storage()),
            1e-4,
            1e-4,
        );

        // Benchmark Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let mut leto_out = leto::Array::zeros([1, axis_n]);
            match op_idx {
                0 => leto_ops::sum_axis_into(
                    black_box(&leto_axis.view()),
                    0,
                    black_box(&mut leto_out.view_mut()),
                )
                .unwrap(),
                1 => leto_ops::min_axis_into(
                    black_box(&leto_axis.view()),
                    0,
                    black_box(&mut leto_out.view_mut()),
                )
                .unwrap(),
                2 => leto_ops::max_axis_into(
                    black_box(&leto_axis.view()),
                    0,
                    black_box(&mut leto_out.view_mut()),
                )
                .unwrap(),
                _ => leto_ops::mean_axis_into(
                    black_box(&leto_axis.view()),
                    0,
                    black_box(&mut leto_out.view_mut()),
                )
                .unwrap(),
            }
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark ndarray
        let t_ndarray = Instant::now();
        for _ in 0..ITERS {
            let _out = match op_idx {
                0 => black_box(&nd_axis).sum_axis(Axis(0)),
                1 => black_box(&nd_axis).map_axis(Axis(0), |v| {
                    *v.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
                }),
                2 => black_box(&nd_axis).map_axis(Axis(0), |v| {
                    *v.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
                }),
                _ => black_box(&nd_axis).mean_axis(Axis(0)).unwrap(),
            };
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_ndarray.elapsed()).as_nanos()
        );

        // Benchmark nalgebra
        let t_nalgebra = Instant::now();
        for _ in 0..ITERS {
            let _out = match op_idx {
                0 => nalgebra_sum_axis(black_box(&na_axis), 0),
                1 => {
                    let mut out = vec![0.0f32; na_axis.ncols()];
                    for c in 0..na_axis.ncols() {
                        let mut m = f32::INFINITY;
                        for r in 0..na_axis.nrows() {
                            m = m.min(na_axis[(r, c)]);
                        }
                        out[c] = m;
                    }
                    out
                }
                2 => {
                    let mut out = vec![0.0f32; na_axis.ncols()];
                    for c in 0..na_axis.ncols() {
                        let mut m = f32::NEG_INFINITY;
                        for r in 0..na_axis.nrows() {
                            m = m.max(na_axis[(r, c)]);
                        }
                        out[c] = m;
                    }
                    out
                }
                _ => {
                    let mut out = vec![0.0f32; na_axis.ncols()];
                    for c in 0..na_axis.ncols() {
                        let mut sum = 0.0f32;
                        for r in 0..na_axis.nrows() {
                            sum += na_axis[(r, c)];
                        }
                        out[c] = sum / na_axis.nrows() as f32;
                    }
                    out
                }
            };
        }
        println!(
            "CPU (nalgebra):{} ns/iter",
            elapsed_per_iter(t_nalgebra.elapsed()).as_nanos()
        );

        // Benchmark CUDA
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            match op_idx {
                0 => sum_axis_into(
                    &cuda_dev,
                    black_box(axis_input),
                    0,
                    black_box(axis_output),
                    BlockWidth::DEFAULT,
                )
                .unwrap(),
                1 => min_axis_into(
                    &cuda_dev,
                    black_box(axis_input),
                    0,
                    black_box(axis_output),
                    BlockWidth::DEFAULT,
                )
                .unwrap(),
                2 => max_axis_into(
                    &cuda_dev,
                    black_box(axis_input),
                    0,
                    black_box(axis_output),
                    BlockWidth::DEFAULT,
                )
                .unwrap(),
                _ => mean_axis_into(
                    &cuda_dev,
                    black_box(axis_input),
                    0,
                    black_box(axis_output),
                    BlockWidth::DEFAULT,
                )
                .unwrap(),
            }
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 5. Matmul (f32, shapes 64x64 and 256x256)
    // ────────────────────────────────────────────────────────────────────────
    for n in &[64, 256] {
        let n = *n;
        println!("--- Benchmarking: Matmul {n}x{n} (f32) ---");
        let host_m: Vec<f32> = (0..(n * n))
            .map(|i| (i as f32 * 0.731 + 1.0) * 1e-4)
            .collect();

        // Leto (CPU)
        let leto_a = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_b = leto_a.clone();
        let mut leto_out = leto::Array::zeros([n, n]);
        leto_ops::matmul(&leto_a.view(), &leto_b.view(), &mut leto_out.view_mut()).unwrap();
        let nd_a = NdArray2::from_shape_vec((n, n), host_m.clone()).unwrap();
        let nd_b = nd_a.clone();
        let na_a = DMatrix::from_vec(n, n, host_m.clone());
        let na_b = na_a.clone();

        // CUDA (GPU)
        let cu_a = cuda_dev.upload(&host_m).unwrap();
        let cu_b = cuda_dev.upload(&host_m).unwrap();
        let cu_out = cuda_dev.alloc_zeroed::<f32>(n * n).unwrap();
        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();
        let op_a = CudaStridedOperand {
            buffer: &cu_a,
            layout: &layout2d,
        };
        let op_b = CudaStridedOperand {
            buffer: &cu_b,
            layout: &layout2d,
        };
        let op_out = CudaStridedOperand {
            buffer: &cu_out,
            layout: &layout2d,
        };
        matmul_into(&cuda_dev, op_a, op_b, op_out).unwrap();
        wait_cuda(&cuda_dev);
        let mut got_cuda = vec![0.0f32; n * n];
        cuda_dev.download(&cu_out, &mut got_cuda).unwrap();
        assert_close_slice(
            &got_cuda,
            leto::Storage::as_slice(leto_out.storage()),
            1e-3,
            1e-3,
        );

        // Benchmark Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let mut leto_out = leto::Array::zeros([n, n]);
            leto_ops::matmul(
                black_box(&leto_a.view()),
                black_box(&leto_b.view()),
                black_box(&mut leto_out.view_mut()),
            )
            .unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark ndarray
        let t_ndarray = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(&nd_a).dot(black_box(&nd_b));
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_ndarray.elapsed()).as_nanos()
        );

        // Benchmark nalgebra
        let t_nalgebra = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(&na_a) * black_box(&na_b);
        }
        println!(
            "CPU (nalgebra):{} ns/iter",
            elapsed_per_iter(t_nalgebra.elapsed()).as_nanos()
        );

        // Benchmark CUDA
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            matmul_into(
                &cuda_dev,
                black_box(op_a),
                black_box(op_b),
                black_box(op_out),
            )
            .unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 6. Cumsum (f32, 256x256 over axis 1)
    // ────────────────────────────────────────────────────────────────────────
    {
        let n = 256;
        println!("--- Benchmarking: Cumsum (f32, {n}x{n} over axis 1) ---");
        let host_m: Vec<f32> = (0..(n * n))
            .map(|i| (i as f32 * 0.731 + 1.0) * 1e-4)
            .collect();

        // Leto
        let leto_a = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let mut leto_out = leto::Array::zeros([n, n]);
        leto_ops::cumsum_into(&leto_a.view(), 1, &mut leto_out.view_mut()).unwrap();
        let nd_a = NdArray2::from_shape_vec((n, n), host_m.clone()).unwrap();
        let na_a = DMatrix::from_vec(n, n, host_m.clone());

        // CUDA
        let cu_a = cuda_dev.upload(&host_m).unwrap();
        let cu_out = cuda_dev.alloc_zeroed::<f32>(n * n).unwrap();
        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();
        let op_a = CudaStridedOperand {
            buffer: &cu_a,
            layout: &layout2d,
        };
        let op_out = CudaStridedOperand {
            buffer: &cu_out,
            layout: &layout2d,
        };
        cumsum_into(&cuda_dev, op_a, 1, op_out, BlockWidth::DEFAULT).unwrap();
        wait_cuda(&cuda_dev);
        let mut got_cuda = vec![0.0f32; n * n];
        cuda_dev.download(&cu_out, &mut got_cuda).unwrap();
        assert_close_slice(
            &got_cuda,
            leto::Storage::as_slice(leto_out.storage()),
            1e-4,
            1e-4,
        );

        // Benchmark Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let mut leto_out = leto::Array::zeros([n, n]);
            leto_ops::cumsum_into(
                black_box(&leto_a.view()),
                1,
                black_box(&mut leto_out.view_mut()),
            )
            .unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark ndarray
        let t_ndarray = Instant::now();
        for _ in 0..ITERS {
            let mut out = nd_a.clone();
            for mut row in out.rows_mut() {
                let mut sum = 0.0f32;
                for val in row.iter_mut() {
                    sum += *val;
                    *val = sum;
                }
            }
            let _ = black_box(out);
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_ndarray.elapsed()).as_nanos()
        );

        // Benchmark nalgebra
        let t_nalgebra = Instant::now();
        for _ in 0..ITERS {
            let mut out = na_a.clone();
            for r in 0..out.nrows() {
                let mut sum = 0.0f32;
                for c in 0..out.ncols() {
                    sum += out[(r, c)];
                    out[(r, c)] = sum;
                }
            }
            let _ = black_box(out);
        }
        println!(
            "CPU (nalgebra):{} ns/iter",
            elapsed_per_iter(t_nalgebra.elapsed()).as_nanos()
        );

        // Benchmark CUDA
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            cumsum_into(
                &cuda_dev,
                black_box(op_a),
                1,
                black_box(op_out),
                BlockWidth::DEFAULT,
            )
            .unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 7. Matrix Power (f32, 64x64 exponent 5)
    // ────────────────────────────────────────────────────────────────────────
    {
        let n = 64;
        println!("--- Benchmarking: Matrix Power (f32, {n}x{n} exponent 5) ---");
        let host_m: Vec<f32> = (0..(n * n))
            .map(|i| (i as f32 * 0.731 + 1.0) * 1e-4)
            .collect();

        // Leto
        let leto_a = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_out = leto_ops::matpow(&leto_a.view(), 5).unwrap();
        let nd_a = NdArray2::from_shape_vec((n, n), host_m.clone()).unwrap();
        let na_a = DMatrix::from_vec(n, n, host_m.clone());

        // CUDA
        let cu_a = cuda_dev.upload(&host_m).unwrap();
        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();
        let op_a = CudaStridedOperand {
            buffer: &cu_a,
            layout: &layout2d,
        };
        let cu_out = matpow(&cuda_dev, op_a, 5).unwrap();
        wait_cuda(&cuda_dev);
        let mut got_cuda = vec![0.0f32; n * n];
        cuda_dev.download(&cu_out, &mut got_cuda).unwrap();
        assert_close_slice(
            &got_cuda,
            leto::Storage::as_slice(leto_out.storage()),
            1e-2,
            1e-2,
        );

        // Benchmark Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _val = leto_ops::matpow(black_box(&leto_a.view()), 5).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark ndarray
        let t_ndarray = Instant::now();
        for _ in 0..ITERS {
            let _out = ndarray_matpow(black_box(&nd_a), 5);
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_ndarray.elapsed()).as_nanos()
        );

        // Benchmark nalgebra
        let t_nalgebra = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(&na_a).pow(5);
        }
        println!(
            "CPU (nalgebra):{} ns/iter",
            elapsed_per_iter(t_nalgebra.elapsed()).as_nanos()
        );

        // Benchmark CUDA
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            let _out = matpow(&cuda_dev, black_box(op_a), 5).unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 8. Kronecker Product (f32, 64x64 ⊗ 8x8)
    // ────────────────────────────────────────────────────────────────────────
    {
        println!("--- Benchmarking: Kronecker Product (f32, 64x64 ⊗ 8x8) ---");
        let a_host: Vec<f32> = (0..(64 * 64))
            .map(|i| (i as f32 * 0.13 + 1.0) * 1e-4)
            .collect();
        let b_host: Vec<f32> = (0..(8 * 8))
            .map(|i| (i as f32 * 0.29 + 2.0) * 1e-4)
            .collect();

        // Leto
        let leto_a = leto::Array::from_shape_vec([64, 64], a_host.clone()).unwrap();
        let leto_b = leto::Array::from_shape_vec([8, 8], b_host.clone()).unwrap();
        let leto_out = leto_ops::kron(&leto_a.view(), &leto_b.view()).unwrap();
        let na_a = DMatrix::from_vec(64, 64, a_host.clone());
        let na_b = DMatrix::from_vec(8, 8, b_host.clone());

        // CUDA
        let cu_a = cuda_dev.upload(&a_host).unwrap();
        let cu_b = cuda_dev.upload(&b_host).unwrap();
        let cu_out = cuda_dev.alloc_zeroed::<f32>(64 * 8 * 64 * 8).unwrap();
        let layout_a = leto::Layout::c_contiguous([64, 64]).unwrap();
        let layout_b = leto::Layout::c_contiguous([8, 8]).unwrap();
        let layout_out = leto::Layout::c_contiguous([64 * 8, 64 * 8]).unwrap();
        let op_a = CudaStridedOperand {
            buffer: &cu_a,
            layout: &layout_a,
        };
        let op_b = CudaStridedOperand {
            buffer: &cu_b,
            layout: &layout_b,
        };
        let op_out = CudaStridedOperand {
            buffer: &cu_out,
            layout: &layout_out,
        };
        kron_into(&cuda_dev, op_a, op_b, op_out).unwrap();
        wait_cuda(&cuda_dev);
        let mut got_cuda = vec![0.0f32; 64 * 8 * 64 * 8];
        cuda_dev.download(&cu_out, &mut got_cuda).unwrap();
        assert_close_slice(
            &got_cuda,
            leto::Storage::as_slice(leto_out.storage()),
            1e-4,
            1e-4,
        );

        // Benchmark Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _val =
                leto_ops::kron(black_box(&leto_a.view()), black_box(&leto_b.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark nalgebra
        let t_nalgebra = Instant::now();
        for _ in 0..ITERS {
            let _out = nalgebra_kron(black_box(&na_a), black_box(&na_b));
        }
        println!(
            "CPU (nalgebra):{} ns/iter",
            elapsed_per_iter(t_nalgebra.elapsed()).as_nanos()
        );

        // Benchmark CUDA
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            kron_into(
                &cuda_dev,
                black_box(op_a),
                black_box(op_b),
                black_box(op_out),
            )
            .unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 9. Dot Product (f32, N=65536)
    // ────────────────────────────────────────────────────────────────────────
    {
        println!("--- Benchmarking: Dot Product (f32, N={LINALG_LEN}) ---");
        let host_a: Vec<f32> = (0..LINALG_LEN)
            .map(|i| (i as f32 * 0.13 + 1.0) * 1e-5)
            .collect();
        let host_b: Vec<f32> = (0..LINALG_LEN)
            .map(|i| (i as f32 * 0.29 + 2.0) * 1e-5)
            .collect();

        // Leto
        let leto_a = leto::Array::from_shape_vec([LINALG_LEN], host_a.clone()).unwrap();
        let leto_b = leto::Array::from_shape_vec([LINALG_LEN], host_b.clone()).unwrap();
        let leto_dot = leto_ops::dot(&leto_a.view(), &leto_b.view()).unwrap();
        let nd_a = NdArray1::from_vec(host_a.clone());
        let nd_b = NdArray1::from_vec(host_b.clone());

        // CUDA
        let cu_a = cuda_dev.upload(&host_a).unwrap();
        let cu_b = cuda_dev.upload(&host_b).unwrap();
        let layout1d = leto::Layout::c_contiguous([LINALG_LEN]).unwrap();
        let cu_dot = dot(
            &cuda_dev,
            CudaStridedOperand {
                buffer: &cu_a,
                layout: &layout1d,
            },
            CudaStridedOperand {
                buffer: &cu_b,
                layout: &layout1d,
            },
        )
        .unwrap();
        let mut got_cuda = vec![0.0f32; 1];
        cuda_dev.download(&cu_dot, &mut got_cuda).unwrap();
        assert!((got_cuda[0] - leto_dot).abs() <= 1e-3);

        // Benchmark Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _val = leto_ops::dot(black_box(&leto_a.view()), black_box(&leto_b.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark ndarray
        let t_ndarray = Instant::now();
        for _ in 0..ITERS {
            let _val = black_box(&nd_a).dot(black_box(&nd_b));
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_ndarray.elapsed()).as_nanos()
        );

        // Benchmark CUDA
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            let _val = dot(
                &cuda_dev,
                black_box(CudaStridedOperand {
                    buffer: &cu_a,
                    layout: &layout1d,
                }),
                black_box(CudaStridedOperand {
                    buffer: &cu_b,
                    layout: &layout1d,
                }),
            )
            .unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 10. Trace (f32, 256x256)
    // ────────────────────────────────────────────────────────────────────────
    {
        let n = 256;
        println!("--- Benchmarking: Trace (f32, {n}x{n}) ---");
        let host_m: Vec<f32> = (0..(n * n))
            .map(|i| (i as f32 * 0.731 + 1.0) * 1e-4)
            .collect();

        // Leto
        let leto_a = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_tr = leto_ops::trace(&leto_a.view()).unwrap();
        let nd_a = NdArray2::from_shape_vec((n, n), host_m.clone()).unwrap();

        // CUDA
        let cu_a = cuda_dev.upload(&host_m).unwrap();
        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();
        let op_a = CudaStridedOperand {
            buffer: &cu_a,
            layout: &layout2d,
        };
        let cu_tr = trace(&cuda_dev, op_a).unwrap();
        let mut got_cuda = vec![0.0f32; 1];
        cuda_dev.download(&cu_tr, &mut got_cuda).unwrap();
        assert!((got_cuda[0] - leto_tr).abs() <= 1e-3);

        // Benchmark Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _val = leto_ops::trace(black_box(&leto_a.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark ndarray
        let t_ndarray = Instant::now();
        for _ in 0..ITERS {
            let mut tr = 0.0f32;
            for i in 0..n {
                tr += nd_a[(i, i)];
            }
            let _ = black_box(tr);
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_ndarray.elapsed()).as_nanos()
        );

        // Benchmark CUDA
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            let _val = trace(&cuda_dev, black_box(op_a)).unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 11. Matrix Rank (f32, 64x64 diagonal rank 32)
    // ────────────────────────────────────────────────────────────────────────
    {
        let n = 64;
        println!("--- Benchmarking: Matrix Rank (f32, {n}x{n} diagonal rank 32) ---");
        let mut host_m = vec![0.0f32; n * n];
        for i in 0..32 {
            host_m[i * n + i] = 1.0f32;
        }

        // Leto
        let leto_a = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_rk = leto_ops::matrix_rank(&leto_a.view()).unwrap();
        assert_eq!(leto_rk, 32);

        // CUDA
        let cu_a = cuda_dev.upload(&host_m).unwrap();
        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();
        let op_a = CudaStridedOperand {
            buffer: &cu_a,
            layout: &layout2d,
        };
        let cu_rk = matrix_rank(&cuda_dev, op_a).unwrap();
        assert_eq!(cu_rk, 32);

        // Benchmark Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _val = leto_ops::matrix_rank(black_box(&leto_a.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark CUDA
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            let _val = matrix_rank(&cuda_dev, black_box(op_a)).unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 12. Determinant (f32, 64x64 diagonal)
    // ────────────────────────────────────────────────────────────────────────
    {
        let n = 64;
        println!("--- Benchmarking: Determinant (f32, {n}x{n} diagonal) ---");
        let mut host_m = vec![0.0f32; n * n];
        for i in 0..n {
            host_m[i * n + i] = 1.01f32;
        }

        // Leto
        let leto_a = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_det = leto_ops::det(&leto_a.view()).unwrap();
        let nd_a = NdArray2::from_shape_vec((n, n), host_m.clone()).unwrap();
        let na_a = DMatrix::from_vec(n, n, host_m.clone());

        // CUDA
        let cu_a = cuda_dev.upload(&host_m).unwrap();
        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();
        let op_a = CudaStridedOperand {
            buffer: &cu_a,
            layout: &layout2d,
        };
        let cu_det = det(&cuda_dev, op_a).unwrap();
        let mut got_cuda = vec![0.0f32; 1];
        cuda_dev.download(&cu_det, &mut got_cuda).unwrap();
        assert!((got_cuda[0] - leto_det).abs() <= 1e-2);

        // Benchmark Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _val = leto_ops::det(black_box(&leto_a.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark ndarray
        let t_ndarray = Instant::now();
        for _ in 0..ITERS {
            let l_mat = nd_a.clone();
            let mut det = 1.0f32;
            for i in 0..n {
                det *= l_mat[(i, i)];
            }
            let _ = black_box(det);
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_ndarray.elapsed()).as_nanos()
        );

        // Benchmark nalgebra
        let t_nalgebra = Instant::now();
        for _ in 0..ITERS {
            let _val = black_box(&na_a).determinant();
        }
        println!(
            "CPU (nalgebra):{} ns/iter",
            elapsed_per_iter(t_nalgebra.elapsed()).as_nanos()
        );

        // Benchmark CUDA
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            let _val = det(&cuda_dev, black_box(op_a)).unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 13. Blocked Cholesky Decomposition (f32, 128x128 SPD)
    // ────────────────────────────────────────────────────────────────────────
    {
        let n = 128;
        println!("--- Benchmarking: Blocked Cholesky Decomposition (f32, {n}x{n} SPD) ---");
        let mut host_m = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                host_m[i * n + j] = if i == j {
                    n as f32 + 5.0f32
                } else {
                    0.02 / (1.0 + (i as f32 - j as f32).abs())
                };
            }
        }

        // Leto
        let leto_a = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_out = leto_ops::cholesky_decompose(&leto_a.view()).unwrap();
        let na_a = DMatrix::from_vec(n, n, host_m.clone());

        // CUDA
        let cu_m = cuda_dev.upload(&host_m).unwrap();
        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();
        let op_a = CudaStridedOperand {
            buffer: &cu_m,
            layout: &layout2d,
        };
        let cu_out = cholesky_decompose_blocked(&cuda_dev, op_a).unwrap();
        wait_cuda(&cuda_dev);
        let mut got_cuda = vec![0.0f32; n * n];
        cuda_dev.download(cu_out.lower(), &mut got_cuda).unwrap();
        assert_close_slice(
            &got_cuda,
            leto::Storage::as_slice(leto_out.lower().storage()),
            1e-4,
            1e-4,
        );

        // Benchmark Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::cholesky_decompose(black_box(&leto_a.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark nalgebra
        let t_nalgebra = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(na_a.clone()).cholesky().unwrap();
        }
        println!(
            "CPU (nalgebra):{} ns/iter",
            elapsed_per_iter(t_nalgebra.elapsed()).as_nanos()
        );

        // Benchmark CUDA
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            let _out = cholesky_decompose_blocked(&cuda_dev, black_box(op_a)).unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 14. LU Decomposition (f32, 32x32)
    // ────────────────────────────────────────────────────────────────────────
    {
        let n = 32;
        println!("--- Benchmarking: LU Decomposition (f32, {n}x{n}) ---");
        let host_m: Vec<f32> = (0..(n * n))
            .map(|i| (i as f32 * 0.731 + 1.0) * 1e-4)
            .collect();

        // Leto
        let leto_a = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_out = leto_ops::lu_decompose(&leto_a.view()).unwrap();
        let na_a = DMatrix::from_vec(n, n, host_m.clone());

        // CUDA
        let cu_m = cuda_dev.upload(&host_m).unwrap();
        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();
        let op_a = CudaStridedOperand {
            buffer: &cu_m,
            layout: &layout2d,
        };
        let cu_out = lu_decompose(&cuda_dev, op_a).unwrap();
        wait_cuda(&cuda_dev);
        let mut got_cuda = vec![0.0f32; n * n];
        cuda_dev.download(cu_out.factors(), &mut got_cuda).unwrap();
        assert_close_slice(
            &got_cuda,
            leto::Storage::as_slice(leto_out.factors().storage()),
            1e-3,
            1e-3,
        );

        // Benchmark Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::lu_decompose(black_box(&leto_a.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark nalgebra
        let t_nalgebra = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(na_a.clone()).lu();
        }
        println!(
            "CPU (nalgebra):{} ns/iter",
            elapsed_per_iter(t_nalgebra.elapsed()).as_nanos()
        );

        // Benchmark CUDA
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            let _out = lu_decompose(&cuda_dev, black_box(op_a)).unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 15. QR Decomposition (f32, 48x24)
    // ────────────────────────────────────────────────────────────────────────
    {
        let rows = 48;
        let cols = 24;
        println!("--- Benchmarking: QR Decomposition (f32, {rows}x{cols}) ---");
        let host_m: Vec<f32> = (0..(rows * cols))
            .map(|i| (i as f32 * 0.731 + 1.0) * 1e-4)
            .collect();

        // Leto
        let leto_a = leto::Array::from_shape_vec([rows, cols], host_m.clone()).unwrap();
        let leto_out = leto_ops::qr_decompose(&leto_a.view()).unwrap();
        let na_a = DMatrix::from_vec(rows, cols, host_m.clone());

        // CUDA
        let cu_m = cuda_dev.upload(&host_m).unwrap();
        let layout2d = leto::Layout::c_contiguous([rows, cols]).unwrap();
        let op_a = CudaStridedOperand {
            buffer: &cu_m,
            layout: &layout2d,
        };
        let cu_out = qr_decompose(&cuda_dev, op_a).unwrap();
        wait_cuda(&cuda_dev);
        let mut got_cuda = vec![0.0f32; rows * cols];
        cuda_dev.download(cu_out.r_buffer(), &mut got_cuda).unwrap();
        assert_close_slice(
            &got_cuda,
            leto::Storage::as_slice(leto_out.r().storage()),
            1e-3,
            1e-3,
        );

        // Benchmark Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::qr_decompose(black_box(&leto_a.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark nalgebra
        let t_nalgebra = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(na_a.clone()).qr();
        }
        println!(
            "CPU (nalgebra):{} ns/iter",
            elapsed_per_iter(t_nalgebra.elapsed()).as_nanos()
        );

        // Benchmark CUDA
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            let _out = qr_decompose(&cuda_dev, black_box(op_a)).unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 15b. Blocked QR Decomposition (f32, 70x35)
    // ────────────────────────────────────────────────────────────────────────
    {
        let rows = 70;
        let cols = 35;
        println!("--- Benchmarking: Blocked QR Decomposition (f32, {rows}x{cols}) ---");
        let host_m: Vec<f32> = (0..(rows * cols))
            .map(|i| (i as f32 * 0.731 + 1.0) * 1e-4)
            .collect();

        // Leto
        let leto_a = leto::Array::from_shape_vec([rows, cols], host_m.clone()).unwrap();
        let leto_out = leto_ops::qr_decompose(&leto_a.view()).unwrap();
        let na_a = DMatrix::from_vec(rows, cols, host_m.clone());

        // CUDA
        let cu_m = cuda_dev.upload(&host_m).unwrap();
        let layout2d = leto::Layout::c_contiguous([rows, cols]).unwrap();
        let op_a = CudaStridedOperand {
            buffer: &cu_m,
            layout: &layout2d,
        };
        let cu_out = qr_decompose_blocked(&cuda_dev, op_a).unwrap();
        wait_cuda(&cuda_dev);
        let mut got_cuda = vec![0.0f32; rows * cols];
        cuda_dev.download(cu_out.r_buffer(), &mut got_cuda).unwrap();
        assert_close_slice(
            &got_cuda,
            leto::Storage::as_slice(leto_out.r().storage()),
            1e-3,
            1e-3,
        );

        // Benchmark Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::qr_decompose(black_box(&leto_a.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark nalgebra
        let t_nalgebra = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(na_a.clone()).qr();
        }
        println!(
            "CPU (nalgebra):{} ns/iter",
            elapsed_per_iter(t_nalgebra.elapsed()).as_nanos()
        );

        // Benchmark CUDA
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            let _out = qr_decompose_blocked(&cuda_dev, black_box(op_a)).unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 16. Symmetric Eigen Jacobi (f32, 32x32)
    // ────────────────────────────────────────────────────────────────────────
    {
        let n = 32;
        println!("--- Benchmarking: Symmetric Eigen Jacobi (f32, {n}x{n}) ---");
        let mut host_m = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                host_m[i * n + j] = if i == j {
                    n as f32 + 5.0f32
                } else {
                    0.02 / (1.0 + (i as f32 - j as f32).abs())
                };
            }
        }

        // Leto
        let leto_a = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_out = leto_ops::symmetric_eigen_jacobi(&leto_a.view()).unwrap();
        let na_a = DMatrix::from_vec(n, n, host_m.clone());

        // CUDA
        let cu_m = cuda_dev.upload(&host_m).unwrap();
        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();
        let op_a = CudaStridedOperand {
            buffer: &cu_m,
            layout: &layout2d,
        };
        let cu_out = symmetric_eigen_jacobi(&cuda_dev, op_a).unwrap();
        wait_cuda(&cuda_dev);
        let mut got_cuda = vec![0.0f32; n];
        cuda_dev
            .download(cu_out.eigenvalues(), &mut got_cuda)
            .unwrap();
        assert_close_slice(&got_cuda, &leto_out.eigenvalues, 1e-3, 1e-3);

        // Benchmark Leto
        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::symmetric_eigen_jacobi(black_box(&leto_a.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        // Benchmark nalgebra
        let t_nalgebra = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(na_a.clone()).symmetric_eigen();
        }
        println!(
            "CPU (nalgebra):{} ns/iter",
            elapsed_per_iter(t_nalgebra.elapsed()).as_nanos()
        );

        // Benchmark CUDA
        let t_cuda = Instant::now();
        for _ in 0..ITERS {
            let _out = symmetric_eigen_jacobi(&cuda_dev, black_box(op_a)).unwrap();
        }
        wait_cuda(&cuda_dev);
        println!(
            "GPU (CUDA):   {} ns/iter\n",
            elapsed_per_iter(t_cuda.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 17. Norms: L1, L2, Max (f32, size N=65536)
    // ────────────────────────────────────────────────────────────────────────
    {
        let host_a: Vec<f32> = (0..LINALG_LEN)
            .map(|i| (i as f32 * 0.13 + 1.0) * 1e-5)
            .collect();

        // Leto
        let leto_a = leto::Array::from_shape_vec([LINALG_LEN], host_a.clone()).unwrap();
        let leto_l1 = leto_ops::norm_l1(&leto_a.view()).unwrap();
        let leto_l2 = leto_ops::norm_l2(&leto_a.view()).unwrap();
        let leto_max = leto_ops::norm_max(&leto_a.view()).unwrap();

        // CUDA
        let cu_a = cuda_dev.upload(&host_a).unwrap();
        let layout = leto::Layout::c_contiguous([LINALG_LEN]).unwrap();
        let op_a = CudaStridedOperand {
            buffer: &cu_a,
            layout: &layout,
        };

        let cu_l1 = norm_l1(&cuda_dev, op_a).unwrap();
        let cu_l2 = norm_l2(&cuda_dev, op_a).unwrap();
        let cu_max = norm_max(&cuda_dev, op_a).unwrap();

        let mut got_l1 = vec![0.0f32; 1];
        let mut got_l2 = vec![0.0f32; 1];
        let mut got_max = vec![0.0f32; 1];

        cuda_dev.download(&cu_l1, &mut got_l1).unwrap();
        cuda_dev.download(&cu_l2, &mut got_l2).unwrap();
        cuda_dev.download(&cu_max, &mut got_max).unwrap();

        assert!((got_l1[0] - leto_l1).abs() <= 1e-3);
        assert!((got_l2[0] - leto_l2).abs() <= 1e-3);
        assert!((got_max[0] - leto_max).abs() <= 1e-3);

        // Benchmark Leto L1, L2, Max
        for op in 0..3 {
            let name = match op {
                0 => "L1",
                1 => "L2",
                _ => "Max",
            };
            println!("--- Benchmarking: Norm {name} (f32, N={LINALG_LEN}) ---");
            let t_leto = Instant::now();
            for _ in 0..ITERS {
                match op {
                    0 => {
                        let _ = leto_ops::norm_l1(black_box(&leto_a.view())).unwrap();
                    }
                    1 => {
                        let _ = leto_ops::norm_l2(black_box(&leto_a.view())).unwrap();
                    }
                    _ => {
                        let _ = leto_ops::norm_max(black_box(&leto_a.view())).unwrap();
                    }
                }
            }
            println!(
                "CPU (Leto):   {} ns/iter",
                elapsed_per_iter(t_leto.elapsed()).as_nanos()
            );

            // Benchmark CUDA
            let t_cuda = Instant::now();
            for _ in 0..ITERS {
                match op {
                    0 => {
                        let _ = norm_l1(&cuda_dev, black_box(op_a)).unwrap();
                    }
                    1 => {
                        let _ = norm_l2(&cuda_dev, black_box(op_a)).unwrap();
                    }
                    _ => {
                        let _ = norm_max(&cuda_dev, black_box(op_a)).unwrap();
                    }
                }
            }
            wait_cuda(&cuda_dev);
            println!(
                "GPU (CUDA):   {} ns/iter\n",
                elapsed_per_iter(t_cuda.elapsed()).as_nanos()
            );
        }
    }
}
