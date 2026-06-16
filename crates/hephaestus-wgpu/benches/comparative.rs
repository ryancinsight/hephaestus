//! Comparative benchmark of Hephaestus GPU (WGPU & CUDA) vs CPU (Leto & ndarray & nalgebra).
//!
//! Validates results across all backends to ensure correctness and outputs timing.

use std::hint::black_box;
use std::time::{Duration, Instant};

use hephaestus_core::BlockWidth;
use hephaestus_cuda::{
    ComputeDevice as CudaComputeDevice, CudaDevice, StridedOperand as CudaStridedOperand,
};
use hephaestus_wgpu::{
    binary_elementwise_into, cholesky_decompose, cumsum_into, det, dot, kron_into, lu_decompose,
    matmul_into, matpow, matrix_rank, max_axis_into, mean_axis_into, min_axis_into, norm_l1,
    norm_l2, norm_max, qr_decompose, reduction, sum_axis_into, trace, unary_elementwise_into,
    AddOp, ExpOp, StridedOperand as WgpuStridedOperand, SumOp, WgpuDevice,
};
use leto::Storage;
use nalgebra::DMatrix;
use ndarray::Array2 as NdArray2;
use ndarray::{Array1 as NdArray1, Axis};

const LEN: usize = 1 << 20; // 1,048,576 elements for elementwise
const LINALG_LEN: usize = 1 << 16; // 65,536 elements for dot/norms
const ITERS: usize = 50;

fn wait_wgpu(device: &WgpuDevice) {
    device
        .inner()
        .poll(wgpu::PollType::Wait)
        .expect("invariant: benchmark device poll succeeds");
}

fn wait_cuda(_device: &CudaDevice) {
    // CUDA launches on the default stream are synchronous or synchronized at download.
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
    match axis {
        0 => (0..matrix.ncols())
            .map(|col| (0..matrix.nrows()).map(|row| matrix[(row, col)]).sum())
            .collect(),
        1 => (0..matrix.nrows())
            .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).sum())
            .collect(),
        _ => panic!("invariant: benchmark axis is rank-2"),
    }
}

fn nalgebra_min_axis(matrix: &DMatrix<f32>, axis: usize) -> Vec<f32> {
    match axis {
        0 => (0..matrix.ncols())
            .map(|col| {
                (0..matrix.nrows())
                    .map(|row| matrix[(row, col)])
                    .fold(f32::INFINITY, f32::min)
            })
            .collect(),
        1 => (0..matrix.nrows())
            .map(|row| {
                (0..matrix.ncols())
                    .map(|col| matrix[(row, col)])
                    .fold(f32::INFINITY, f32::min)
            })
            .collect(),
        _ => panic!("invariant: benchmark axis is rank-2"),
    }
}

fn nalgebra_max_axis(matrix: &DMatrix<f32>, axis: usize) -> Vec<f32> {
    match axis {
        0 => (0..matrix.ncols())
            .map(|col| {
                (0..matrix.nrows())
                    .map(|row| matrix[(row, col)])
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect(),
        1 => (0..matrix.nrows())
            .map(|row| {
                (0..matrix.ncols())
                    .map(|col| matrix[(row, col)])
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect(),
        _ => panic!("invariant: benchmark axis is rank-2"),
    }
}

fn nalgebra_mean_axis(matrix: &DMatrix<f32>, axis: usize) -> Vec<f32> {
    let mut out = nalgebra_sum_axis(matrix, axis);
    let denominator = match axis {
        0 => matrix.nrows() as f32,
        1 => matrix.ncols() as f32,
        _ => panic!("invariant: benchmark axis is rank-2"),
    };
    for value in &mut out {
        *value /= denominator;
    }
    out
}

fn ndarray_cumsum_axis(matrix: &NdArray2<f32>, axis: usize) -> NdArray2<f32> {
    let mut out = matrix.clone();
    match axis {
        0 => {
            for row in 1..matrix.nrows() {
                for col in 0..matrix.ncols() {
                    out[(row, col)] += out[(row - 1, col)];
                }
            }
        }
        1 => {
            for row in 0..matrix.nrows() {
                for col in 1..matrix.ncols() {
                    out[(row, col)] += out[(row, col - 1)];
                }
            }
        }
        _ => panic!("invariant: benchmark axis is rank-2"),
    }
    out
}

fn nalgebra_cumsum_axis(matrix: &DMatrix<f32>, axis: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; matrix.nrows() * matrix.ncols()];
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            out[row * matrix.ncols() + col] = matrix[(row, col)];
        }
    }
    match axis {
        0 => {
            for row in 1..matrix.nrows() {
                for col in 0..matrix.ncols() {
                    let i = row * matrix.ncols() + col;
                    let previous = (row - 1) * matrix.ncols() + col;
                    out[i] += out[previous];
                }
            }
        }
        1 => {
            for row in 0..matrix.nrows() {
                for col in 1..matrix.ncols() {
                    let i = row * matrix.ncols() + col;
                    out[i] += out[i - 1];
                }
            }
        }
        _ => panic!("invariant: benchmark axis is rank-2"),
    }
    out
}

fn main() {
    println!("=== Starting comparative benchmarks (GPU vs CPU) ===");
    println!("Iterations: {ITERS}");

    // Setup WGPU
    let wgpu_dev = match WgpuDevice::try_default("hephaestus-comparative-bench") {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping benchmarks: WGPU device unavailable: {e}");
            return;
        }
    };
    println!("WGPU GPU Backend: {}", wgpu_dev.backend_name());

    // Setup CUDA
    let cuda_dev = match CudaDevice::try_default() {
        Ok(d) => {
            println!("CUDA GPU Backend acquired successfully.\n");
            Some(d)
        }
        Err(e) => {
            println!("CUDA GPU Backend not available (skipping CUDA benchmarks): {e}\n");
            None
        }
    };

    // ────────────────────────────────────────────────────────────────────────
    // 1. Elementwise Add (f32, size 1<<20)
    // ────────────────────────────────────────────────────────────────────────
    {
        println!("--- Benchmarking: Elementwise Add (f32, N={LEN}) ---");
        let host_a: Vec<f32> = (0..LEN).map(|i| (i as f32 * 0.731 + 1.0) * 1e-7).collect();
        let host_b: Vec<f32> = (0..LEN).map(|i| (i as f32 * 0.317 + 2.0) * 1e-7).collect();

        // CPU Leto & ndarray
        let leto_a = leto::Array::from_shape_vec([LEN], host_a.clone()).unwrap();
        let leto_b = leto::Array::from_shape_vec([LEN], host_b.clone()).unwrap();
        let mut leto_out = leto::Array::zeros([LEN]);
        leto_ops::add(&leto_a.view(), &leto_b.view(), &mut leto_out.view_mut()).unwrap();
        let nd_a = NdArray1::from_vec(host_a.clone());
        let nd_b = NdArray1::from_vec(host_b.clone());

        // WGPU
        let wg_a = wgpu_dev.upload(&host_a).unwrap();
        let wg_b = wgpu_dev.upload(&host_b).unwrap();
        let wg_out = wgpu_dev.alloc_zeroed::<f32>(LEN).unwrap();
        binary_elementwise_into::<AddOp, f32>(
            &wgpu_dev,
            &wg_a,
            &wg_b,
            &wg_out,
            BlockWidth::DEFAULT,
        )
        .unwrap();
        wait_wgpu(&wgpu_dev);
        let mut got_wgpu = vec![0.0f32; LEN];
        wgpu_dev.download(&wg_out, &mut got_wgpu).unwrap();
        assert_close_slice(&got_wgpu, leto_out.storage().as_slice(), 1e-5, 0.0);

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            binary_elementwise_into::<AddOp, f32>(
                &wgpu_dev,
                &wg_a,
                &wg_b,
                &wg_out,
                BlockWidth::DEFAULT,
            )
            .unwrap();
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        // CUDA
        if let Some(ref cuda) = cuda_dev {
            let cu_a = cuda.upload(&host_a).unwrap();
            let cu_b = cuda.upload(&host_b).unwrap();
            let cu_out = cuda.alloc_zeroed::<f32>(LEN).unwrap();
            hephaestus_cuda::binary_elementwise_into::<hephaestus_cuda::AddOp, f32>(
                cuda,
                &cu_a,
                &cu_b,
                &cu_out,
                BlockWidth::DEFAULT,
            )
            .unwrap();
            let mut got_cuda = vec![0.0f32; LEN];
            cuda.download(&cu_out, &mut got_cuda).unwrap();
            assert_close_slice(&got_cuda, leto_out.storage().as_slice(), 1e-5, 0.0);

            let t_cuda = Instant::now();
            for _ in 0..ITERS {
                hephaestus_cuda::binary_elementwise_into::<hephaestus_cuda::AddOp, f32>(
                    cuda,
                    &cu_a,
                    &cu_b,
                    &cu_out,
                    BlockWidth::DEFAULT,
                )
                .unwrap();
            }
            wait_cuda(cuda);
            println!(
                "GPU (CUDA):   {} ns/iter",
                elapsed_per_iter(t_cuda.elapsed()).as_nanos()
            );
        }

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let mut out = black_box(leto::Array::zeros([LEN]));
            leto_ops::add(
                black_box(&leto_a.view()),
                black_box(&leto_b.view()),
                &mut out.view_mut(),
            )
            .unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        let t_nd = Instant::now();
        for _ in 0..ITERS {
            let _c = black_box(&nd_a) + black_box(&nd_b);
        }
        println!(
            "CPU (ndarray):{} ns/iter\n",
            elapsed_per_iter(t_nd.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 2. Elementwise Exp (f32, size 1<<20)
    // ────────────────────────────────────────────────────────────────────────
    {
        println!("--- Benchmarking: Elementwise Exp (f32, N={LEN}) ---");
        let host_a: Vec<f32> = (0..LEN).map(|i| (i as f32 * 0.731 + 1.0) * 1e-7).collect();

        // CPU Leto & ndarray
        let leto_a = leto::Array::from_shape_vec([LEN], host_a.clone()).unwrap();
        let leto_out = leto_ops::unary_map(leto_ops::ExpOp, &leto_a.view()).unwrap();
        let nd_a = NdArray1::from_vec(host_a.clone());

        // WGPU
        let wg_a = wgpu_dev.upload(&host_a).unwrap();
        let wg_out = wgpu_dev.alloc_zeroed::<f32>(LEN).unwrap();
        unary_elementwise_into::<ExpOp, f32>(&wgpu_dev, &wg_a, &wg_out, BlockWidth::DEFAULT)
            .unwrap();
        wait_wgpu(&wgpu_dev);
        let mut got_wgpu = vec![0.0f32; LEN];
        wgpu_dev.download(&wg_out, &mut got_wgpu).unwrap();
        assert_close_slice(&got_wgpu, leto_out.storage().as_slice(), 1e-5, 0.0);

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            unary_elementwise_into::<ExpOp, f32>(&wgpu_dev, &wg_a, &wg_out, BlockWidth::DEFAULT)
                .unwrap();
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        // CUDA
        if let Some(ref cuda) = cuda_dev {
            let cu_a = cuda.upload(&host_a).unwrap();
            let cu_out = cuda.alloc_zeroed::<f32>(LEN).unwrap();
            hephaestus_cuda::unary_elementwise_into::<hephaestus_cuda::ExpOp, f32>(
                cuda,
                &cu_a,
                &cu_out,
                BlockWidth::DEFAULT,
            )
            .unwrap();
            let mut got_cuda = vec![0.0f32; LEN];
            cuda.download(&cu_out, &mut got_cuda).unwrap();
            assert_close_slice(&got_cuda, leto_out.storage().as_slice(), 1e-5, 0.0);

            let t_cuda = Instant::now();
            for _ in 0..ITERS {
                hephaestus_cuda::unary_elementwise_into::<hephaestus_cuda::ExpOp, f32>(
                    cuda,
                    &cu_a,
                    &cu_out,
                    BlockWidth::DEFAULT,
                )
                .unwrap();
            }
            wait_cuda(cuda);
            println!(
                "GPU (CUDA):   {} ns/iter",
                elapsed_per_iter(t_cuda.elapsed()).as_nanos()
            );
        }

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::unary_map(leto_ops::ExpOp, black_box(&leto_a.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        let t_nd = Instant::now();
        for _ in 0..ITERS {
            let _c = black_box(&nd_a).mapv(f32::exp);
        }
        println!(
            "CPU (ndarray):{} ns/iter\n",
            elapsed_per_iter(t_nd.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 3. Sum Reduction (f32, size 1<<20)
    // ────────────────────────────────────────────────────────────────────────
    {
        println!("--- Benchmarking: Sum Reduction (f32, N={LEN}) ---");
        let host_a: Vec<f32> = (0..LEN).map(|i| (i as f32 * 0.731 + 1.0) * 1e-7).collect();

        // CPU Leto & ndarray
        let leto_a = leto::Array::from_shape_vec([LEN], host_a.clone()).unwrap();
        let leto_out = leto_ops::sum(&leto_a.view());
        let nd_a = NdArray1::from_vec(host_a.clone());

        // WGPU
        let wg_a = wgpu_dev.upload(&host_a).unwrap();
        let wg_out = reduction::<SumOp, f32>(&wgpu_dev, &wg_a).unwrap();
        wait_wgpu(&wgpu_dev);
        let mut got_wgpu = [0.0f32; 1];
        wgpu_dev.download(&wg_out, &mut got_wgpu).unwrap();
        assert!((got_wgpu[0] - leto_out).abs() < 1e-2 * leto_out.abs());

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            let _res = reduction::<SumOp, f32>(&wgpu_dev, &wg_a).unwrap();
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        // CUDA
        if let Some(ref cuda) = cuda_dev {
            let cu_a = cuda.upload(&host_a).unwrap();
            let cu_out =
                hephaestus_cuda::reduction::<hephaestus_cuda::SumOp, f32>(cuda, &cu_a).unwrap();
            let mut got_cuda = [0.0f32; 1];
            cuda.download(&cu_out, &mut got_cuda).unwrap();
            assert!((got_cuda[0] - leto_out).abs() < 1e-2 * leto_out.abs());

            let t_cuda = Instant::now();
            for _ in 0..ITERS {
                let _res =
                    hephaestus_cuda::reduction::<hephaestus_cuda::SumOp, f32>(cuda, &cu_a).unwrap();
            }
            wait_cuda(cuda);
            println!(
                "GPU (CUDA):   {} ns/iter",
                elapsed_per_iter(t_cuda.elapsed()).as_nanos()
            );
        }

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _val = leto_ops::sum(black_box(&leto_a.view()));
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        let t_nd = Instant::now();
        for _ in 0..ITERS {
            let _val = black_box(&nd_a).sum();
        }
        println!(
            "CPU (ndarray):{} ns/iter\n",
            elapsed_per_iter(t_nd.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 4. Axis reductions (Sum, Min, Max, Mean) (256x256, axis 0)
    // ────────────────────────────────────────────────────────────────────────
    let axis_n = 256usize;
    let host_axis: Vec<f32> = (0..axis_n * axis_n)
        .map(|i| (i as f32 * 0.43 + 1.0) * 1e-4)
        .collect();

    let leto_axis = leto::Array::from_shape_vec([axis_n, axis_n], host_axis.clone()).unwrap();
    let nd_axis = NdArray2::from_shape_vec([axis_n, axis_n], host_axis.clone()).unwrap();
    let na_axis = DMatrix::from_row_slice(axis_n, axis_n, &host_axis);

    let axis_input_layout = leto::Layout::c_contiguous([axis_n, axis_n]).unwrap();
    let axis_output_layout = leto::Layout::c_contiguous([1, axis_n]).unwrap();

    let ops = vec![
        ("Sum", 0),  // 0: sum
        ("Min", 1),  // 1: min
        ("Max", 2),  // 2: max
        ("Mean", 3), // 3: mean
    ];

    for (name, op_idx) in ops {
        println!("--- Benchmarking: Axis {name} (f32, 256x256 over axis 0) ---");

        // CPU Leto
        let leto_out = match op_idx {
            0 => leto_ops::sum_axis(&leto_axis.view(), 0).unwrap(),
            1 => leto_ops::min_axis(&leto_axis.view(), 0).unwrap(),
            2 => leto_ops::max_axis(&leto_axis.view(), 0).unwrap(),
            _ => leto_ops::mean_axis(&leto_axis.view(), 0).unwrap(),
        };

        // WGPU
        let wg_axis = wgpu_dev.upload(&host_axis).unwrap();
        let wg_out = wgpu_dev.alloc_zeroed::<f32>(axis_n).unwrap();
        let axis_input = WgpuStridedOperand {
            buffer: &wg_axis,
            layout: &axis_input_layout,
        };
        let axis_output = WgpuStridedOperand {
            buffer: &wg_out,
            layout: &axis_output_layout,
        };

        match op_idx {
            0 => sum_axis_into(&wgpu_dev, axis_input, 0, axis_output, BlockWidth::DEFAULT).unwrap(),
            1 => min_axis_into(&wgpu_dev, axis_input, 0, axis_output, BlockWidth::DEFAULT).unwrap(),
            2 => max_axis_into(&wgpu_dev, axis_input, 0, axis_output, BlockWidth::DEFAULT).unwrap(),
            _ => {
                mean_axis_into(&wgpu_dev, axis_input, 0, axis_output, BlockWidth::DEFAULT).unwrap()
            }
        }
        wait_wgpu(&wgpu_dev);
        let mut got_wgpu = vec![0.0f32; axis_n];
        wgpu_dev.download(&wg_out, &mut got_wgpu).unwrap();
        assert_close_slice(
            &got_wgpu,
            leto_out.storage().as_slice(),
            0.0,
            4.0 * f32::EPSILON,
        );

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            match op_idx {
                0 => sum_axis_into(&wgpu_dev, axis_input, 0, axis_output, BlockWidth::DEFAULT)
                    .unwrap(),
                1 => min_axis_into(&wgpu_dev, axis_input, 0, axis_output, BlockWidth::DEFAULT)
                    .unwrap(),
                2 => max_axis_into(&wgpu_dev, axis_input, 0, axis_output, BlockWidth::DEFAULT)
                    .unwrap(),
                _ => mean_axis_into(&wgpu_dev, axis_input, 0, axis_output, BlockWidth::DEFAULT)
                    .unwrap(),
            }
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        // CUDA
        if let Some(ref cuda) = cuda_dev {
            let cu_axis = cuda.upload(&host_axis).unwrap();
            let cu_out = cuda.alloc_zeroed::<f32>(axis_n).unwrap();
            let cu_input = CudaStridedOperand {
                buffer: &cu_axis,
                layout: &axis_input_layout,
            };
            let cu_output = CudaStridedOperand {
                buffer: &cu_out,
                layout: &axis_output_layout,
            };

            match op_idx {
                0 => hephaestus_cuda::sum_axis_into(
                    cuda,
                    cu_input,
                    0,
                    cu_output,
                    BlockWidth::DEFAULT,
                )
                .unwrap(),
                1 => hephaestus_cuda::min_axis_into(
                    cuda,
                    cu_input,
                    0,
                    cu_output,
                    BlockWidth::DEFAULT,
                )
                .unwrap(),
                2 => hephaestus_cuda::max_axis_into(
                    cuda,
                    cu_input,
                    0,
                    cu_output,
                    BlockWidth::DEFAULT,
                )
                .unwrap(),
                _ => hephaestus_cuda::mean_axis_into(
                    cuda,
                    cu_input,
                    0,
                    cu_output,
                    BlockWidth::DEFAULT,
                )
                .unwrap(),
            }
            wait_cuda(cuda);
            let mut got_cuda = vec![0.0f32; axis_n];
            cuda.download(&cu_out, &mut got_cuda).unwrap();
            assert_close_slice(
                &got_cuda,
                leto_out.storage().as_slice(),
                0.0,
                4.0 * f32::EPSILON,
            );

            let t_cuda = Instant::now();
            for _ in 0..ITERS {
                match op_idx {
                    0 => hephaestus_cuda::sum_axis_into(
                        cuda,
                        cu_input,
                        0,
                        cu_output,
                        BlockWidth::DEFAULT,
                    )
                    .unwrap(),
                    1 => hephaestus_cuda::min_axis_into(
                        cuda,
                        cu_input,
                        0,
                        cu_output,
                        BlockWidth::DEFAULT,
                    )
                    .unwrap(),
                    2 => hephaestus_cuda::max_axis_into(
                        cuda,
                        cu_input,
                        0,
                        cu_output,
                        BlockWidth::DEFAULT,
                    )
                    .unwrap(),
                    _ => hephaestus_cuda::mean_axis_into(
                        cuda,
                        cu_input,
                        0,
                        cu_output,
                        BlockWidth::DEFAULT,
                    )
                    .unwrap(),
                }
            }
            wait_cuda(cuda);
            println!(
                "GPU (CUDA):   {} ns/iter",
                elapsed_per_iter(t_cuda.elapsed()).as_nanos()
            );
        }

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = match op_idx {
                0 => leto_ops::sum_axis(black_box(&leto_axis.view()), 0).unwrap(),
                1 => leto_ops::min_axis(black_box(&leto_axis.view()), 0).unwrap(),
                2 => leto_ops::max_axis(black_box(&leto_axis.view()), 0).unwrap(),
                _ => leto_ops::mean_axis(black_box(&leto_axis.view()), 0).unwrap(),
            };
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        let t_nd = Instant::now();
        for _ in 0..ITERS {
            let _out = match op_idx {
                0 => black_box(&nd_axis).sum_axis(Axis(0)).to_vec(),
                1 => black_box(&nd_axis)
                    .fold_axis(Axis(0), f32::INFINITY, |acc, x| acc.min(*x))
                    .to_vec(),
                2 => black_box(&nd_axis)
                    .fold_axis(Axis(0), f32::NEG_INFINITY, |acc, x| acc.max(*x))
                    .to_vec(),
                _ => black_box(&nd_axis).mean_axis(Axis(0)).unwrap().to_vec(),
            };
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_nd.elapsed()).as_nanos()
        );

        let t_na = Instant::now();
        for _ in 0..ITERS {
            let _out = match op_idx {
                0 => nalgebra_sum_axis(black_box(&na_axis), 0),
                1 => nalgebra_min_axis(black_box(&na_axis), 0),
                2 => nalgebra_max_axis(black_box(&na_axis), 0),
                _ => nalgebra_mean_axis(black_box(&na_axis), 0),
            };
        }
        println!(
            "CPU (nalgebra):{} ns/iter\n",
            elapsed_per_iter(t_na.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 5. Matmul 64x64 and 256x256
    // ────────────────────────────────────────────────────────────────────────
    for n in [64usize, 256usize] {
        println!("--- Benchmarking: Matmul {n}x{n} (f32) ---");
        let host_m: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.17 + 1.0) * 1e-3).collect();

        // CPU Leto & ndarray & nalgebra
        let leto_m = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let mut leto_out = leto::Array::zeros([n, n]);
        leto_ops::matmul(&leto_m.view(), &leto_m.view(), &mut leto_out.view_mut()).unwrap();
        let nd_m = NdArray2::from_shape_vec([n, n], host_m.clone()).unwrap();
        let na_m = DMatrix::from_row_slice(n, n, &host_m);

        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();

        // WGPU
        let wg_m = wgpu_dev.upload(&host_m).unwrap();
        let wg_out = wgpu_dev.alloc_zeroed::<f32>(n * n).unwrap();
        matmul_into(
            &wgpu_dev,
            WgpuStridedOperand {
                buffer: &wg_m,
                layout: &layout2d,
            },
            WgpuStridedOperand {
                buffer: &wg_m,
                layout: &layout2d,
            },
            WgpuStridedOperand {
                buffer: &wg_out,
                layout: &layout2d,
            },
        )
        .unwrap();
        wait_wgpu(&wgpu_dev);
        let mut got_wgpu = vec![0.0f32; n * n];
        wgpu_dev.download(&wg_out, &mut got_wgpu).unwrap();
        assert_close_slice(&got_wgpu, leto_out.storage().as_slice(), 0.0, 1e-4);

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            matmul_into(
                &wgpu_dev,
                WgpuStridedOperand {
                    buffer: &wg_m,
                    layout: &layout2d,
                },
                WgpuStridedOperand {
                    buffer: &wg_m,
                    layout: &layout2d,
                },
                WgpuStridedOperand {
                    buffer: &wg_out,
                    layout: &layout2d,
                },
            )
            .unwrap();
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        // CUDA
        if let Some(ref cuda) = cuda_dev {
            let cu_m = cuda.upload(&host_m).unwrap();
            let cu_out = cuda.alloc_zeroed::<f32>(n * n).unwrap();
            hephaestus_cuda::matmul_into(
                cuda,
                CudaStridedOperand {
                    buffer: &cu_m,
                    layout: &layout2d,
                },
                CudaStridedOperand {
                    buffer: &cu_m,
                    layout: &layout2d,
                },
                CudaStridedOperand {
                    buffer: &cu_out,
                    layout: &layout2d,
                },
            )
            .unwrap();
            wait_cuda(cuda);
            let mut got_cuda = vec![0.0f32; n * n];
            cuda.download(&cu_out, &mut got_cuda).unwrap();
            assert_close_slice(&got_cuda, leto_out.storage().as_slice(), 0.0, 1e-4);

            let t_cuda = Instant::now();
            for _ in 0..ITERS {
                hephaestus_cuda::matmul_into(
                    cuda,
                    CudaStridedOperand {
                        buffer: &cu_m,
                        layout: &layout2d,
                    },
                    CudaStridedOperand {
                        buffer: &cu_m,
                        layout: &layout2d,
                    },
                    CudaStridedOperand {
                        buffer: &cu_out,
                        layout: &layout2d,
                    },
                )
                .unwrap();
            }
            wait_cuda(cuda);
            println!(
                "GPU (CUDA):   {} ns/iter",
                elapsed_per_iter(t_cuda.elapsed()).as_nanos()
            );
        }

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let mut out = black_box(leto::Array::zeros([n, n]));
            leto_ops::matmul(
                black_box(&leto_m.view()),
                black_box(&leto_m.view()),
                &mut out.view_mut(),
            )
            .unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        let t_nd = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(&nd_m).dot(black_box(&nd_m));
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_nd.elapsed()).as_nanos()
        );

        let t_na = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(&na_m).pow(5); // nalgebra pow is on DMatrix, wait: is it na_m * na_m or pow? In original it was * na_m, let's keep * na_m
        }
        println!(
            "CPU (nalgebra):{} ns/iter\n",
            elapsed_per_iter(t_na.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 6. Cumsum (256x256, axis 1)
    // ────────────────────────────────────────────────────────────────────────
    {
        let n = 256usize;
        println!("--- Benchmarking: Cumsum (f32, {n}x{n} over axis 1) ---");
        let host_m: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.17 + 1.0) * 1e-4).collect();

        // CPU Leto & ndarray & nalgebra
        let leto_m = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_out = leto_ops::cumsum(&leto_m.view(), 1).unwrap();
        let nd_m = NdArray2::from_shape_vec([n, n], host_m.clone()).unwrap();
        let na_m = DMatrix::from_row_slice(n, n, &host_m);

        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();

        // WGPU
        let wg_m = wgpu_dev.upload(&host_m).unwrap();
        let wg_out = wgpu_dev.alloc_zeroed::<f32>(n * n).unwrap();
        cumsum_into(
            &wgpu_dev,
            WgpuStridedOperand {
                buffer: &wg_m,
                layout: &layout2d,
            },
            1,
            WgpuStridedOperand {
                buffer: &wg_out,
                layout: &layout2d,
            },
            BlockWidth::DEFAULT,
        )
        .unwrap();
        wait_wgpu(&wgpu_dev);
        let mut got_wgpu = vec![0.0f32; n * n];
        wgpu_dev.download(&wg_out, &mut got_wgpu).unwrap();
        assert_close_slice(&got_wgpu, leto_out.storage().as_slice(), 0.0, 1e-4);

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            cumsum_into(
                &wgpu_dev,
                WgpuStridedOperand {
                    buffer: &wg_m,
                    layout: &layout2d,
                },
                1,
                WgpuStridedOperand {
                    buffer: &wg_out,
                    layout: &layout2d,
                },
                BlockWidth::DEFAULT,
            )
            .unwrap();
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        // CUDA
        if let Some(ref cuda) = cuda_dev {
            let cu_m = cuda.upload(&host_m).unwrap();
            let cu_out = cuda.alloc_zeroed::<f32>(n * n).unwrap();
            hephaestus_cuda::cumsum_into(
                cuda,
                CudaStridedOperand {
                    buffer: &cu_m,
                    layout: &layout2d,
                },
                1,
                CudaStridedOperand {
                    buffer: &cu_out,
                    layout: &layout2d,
                },
                BlockWidth::DEFAULT,
            )
            .unwrap();
            wait_cuda(cuda);
            let mut got_cuda = vec![0.0f32; n * n];
            cuda.download(&cu_out, &mut got_cuda).unwrap();
            assert_close_slice(&got_cuda, leto_out.storage().as_slice(), 0.0, 1e-4);

            let t_cuda = Instant::now();
            for _ in 0..ITERS {
                hephaestus_cuda::cumsum_into(
                    cuda,
                    CudaStridedOperand {
                        buffer: &cu_m,
                        layout: &layout2d,
                    },
                    1,
                    CudaStridedOperand {
                        buffer: &cu_out,
                        layout: &layout2d,
                    },
                    BlockWidth::DEFAULT,
                )
                .unwrap();
            }
            wait_cuda(cuda);
            println!(
                "GPU (CUDA):   {} ns/iter",
                elapsed_per_iter(t_cuda.elapsed()).as_nanos()
            );
        }

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::cumsum(black_box(&leto_m.view()), 1).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        let t_nd = Instant::now();
        for _ in 0..ITERS {
            let _out = ndarray_cumsum_axis(black_box(&nd_m), 1);
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_nd.elapsed()).as_nanos()
        );

        let t_na = Instant::now();
        for _ in 0..ITERS {
            let _out = nalgebra_cumsum_axis(black_box(&na_m), 1);
        }
        println!(
            "CPU (nalgebra):{} ns/iter\n",
            elapsed_per_iter(t_na.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 7. Matrix Power (f32, 64x64, exponent 5)
    // ────────────────────────────────────────────────────────────────────────
    {
        let n = 64usize;
        println!("--- Benchmarking: Matrix Power (f32, {n}x{n} exponent 5) ---");
        let host_m: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.17 + 1.0) * 1e-3).collect();

        // CPU Leto & ndarray & nalgebra
        let leto_m = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_out = leto_ops::matpow(&leto_m.view(), 5).unwrap();
        let nd_m = NdArray2::from_shape_vec([n, n], host_m.clone()).unwrap();
        let na_m = DMatrix::from_row_slice(n, n, &host_m);

        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();

        // WGPU
        let wg_m = wgpu_dev.upload(&host_m).unwrap();
        let wg_out = matpow(
            &wgpu_dev,
            WgpuStridedOperand {
                buffer: &wg_m,
                layout: &layout2d,
            },
            5,
        )
        .unwrap();
        wait_wgpu(&wgpu_dev);
        let mut got_wgpu = vec![0.0f32; n * n];
        wgpu_dev.download(&wg_out, &mut got_wgpu).unwrap();
        assert_close_slice(&got_wgpu, leto_out.storage().as_slice(), 0.0, 1e-3);

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            let _out = matpow(
                &wgpu_dev,
                WgpuStridedOperand {
                    buffer: &wg_m,
                    layout: &layout2d,
                },
                5,
            )
            .unwrap();
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        // CUDA
        if let Some(ref cuda) = cuda_dev {
            let cu_m = cuda.upload(&host_m).unwrap();
            let cu_out = hephaestus_cuda::matpow(
                cuda,
                CudaStridedOperand {
                    buffer: &cu_m,
                    layout: &layout2d,
                },
                5,
            )
            .unwrap();
            wait_cuda(cuda);
            let mut got_cuda = vec![0.0f32; n * n];
            cuda.download(&cu_out, &mut got_cuda).unwrap();
            assert_close_slice(&got_cuda, leto_out.storage().as_slice(), 0.0, 1e-3);

            let t_cuda = Instant::now();
            for _ in 0..ITERS {
                let _out = hephaestus_cuda::matpow(
                    cuda,
                    CudaStridedOperand {
                        buffer: &cu_m,
                        layout: &layout2d,
                    },
                    5,
                )
                .unwrap();
            }
            wait_cuda(cuda);
            println!(
                "GPU (CUDA):   {} ns/iter",
                elapsed_per_iter(t_cuda.elapsed()).as_nanos()
            );
        }

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::matpow(black_box(&leto_m.view()), 5).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        let t_nd = Instant::now();
        for _ in 0..ITERS {
            let _out = ndarray_matpow(black_box(&nd_m), 5);
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_nd.elapsed()).as_nanos()
        );

        let t_na = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(&na_m).pow(5);
        }
        println!(
            "CPU (nalgebra):{} ns/iter\n",
            elapsed_per_iter(t_na.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 8. Kronecker Product (f32, 64x64 ⊗ 8x8)
    // ────────────────────────────────────────────────────────────────────────
    {
        println!("--- Benchmarking: Kronecker Product (f32, 64x64 ⊗ 8x8) ---");
        let a_n = 64usize;
        let b_n = 8usize;
        let host_a: Vec<f32> = (0..a_n * a_n)
            .map(|i| (i as f32 * 0.17 + 1.0) * 1e-4)
            .collect();
        let host_b: Vec<f32> = (0..b_n * b_n)
            .map(|i| (i as f32 * 0.31 + 2.0) * 1e-4)
            .collect();

        // CPU Leto & ndarray & nalgebra
        let leto_a = leto::Array::from_shape_vec([a_n, a_n], host_a.clone()).unwrap();
        let leto_b = leto::Array::from_shape_vec([b_n, b_n], host_b.clone()).unwrap();
        let leto_out = leto_ops::kron(&leto_a.view(), &leto_b.view()).unwrap();
        let na_a = DMatrix::from_row_slice(a_n, a_n, &host_a);
        let na_b = DMatrix::from_row_slice(b_n, b_n, &host_b);

        let layout_a = leto::Layout::c_contiguous([a_n, a_n]).unwrap();
        let layout_b = leto::Layout::c_contiguous([b_n, b_n]).unwrap();

        // WGPU
        let wg_a = wgpu_dev.upload(&host_a).unwrap();
        let wg_b = wgpu_dev.upload(&host_b).unwrap();
        let wg_out = wgpu_dev.alloc_zeroed::<f32>(a_n * b_n * a_n * b_n).unwrap();
        let out_layout = leto::Layout::c_contiguous([a_n * b_n, a_n * b_n]).unwrap();
        kron_into(
            &wgpu_dev,
            WgpuStridedOperand {
                buffer: &wg_a,
                layout: &layout_a,
            },
            WgpuStridedOperand {
                buffer: &wg_b,
                layout: &layout_b,
            },
            WgpuStridedOperand {
                buffer: &wg_out,
                layout: &out_layout,
            },
        )
        .unwrap();
        wait_wgpu(&wgpu_dev);
        let mut got_wgpu = vec![0.0f32; a_n * b_n * a_n * b_n];
        wgpu_dev.download(&wg_out, &mut got_wgpu).unwrap();
        assert_close_slice(&got_wgpu, leto_out.storage().as_slice(), 0.0, 1e-5);

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            kron_into(
                &wgpu_dev,
                WgpuStridedOperand {
                    buffer: &wg_a,
                    layout: &layout_a,
                },
                WgpuStridedOperand {
                    buffer: &wg_b,
                    layout: &layout_b,
                },
                WgpuStridedOperand {
                    buffer: &wg_out,
                    layout: &out_layout,
                },
            )
            .unwrap();
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        // CUDA
        if let Some(ref cuda) = cuda_dev {
            let cu_a = cuda.upload(&host_a).unwrap();
            let cu_b = cuda.upload(&host_b).unwrap();
            let cu_out = cuda.alloc_zeroed::<f32>(a_n * b_n * a_n * b_n).unwrap();
            hephaestus_cuda::kron_into(
                cuda,
                CudaStridedOperand {
                    buffer: &cu_a,
                    layout: &layout_a,
                },
                CudaStridedOperand {
                    buffer: &cu_b,
                    layout: &layout_b,
                },
                CudaStridedOperand {
                    buffer: &cu_out,
                    layout: &out_layout,
                },
            )
            .unwrap();
            wait_cuda(cuda);
            let mut got_cuda = vec![0.0f32; a_n * b_n * a_n * b_n];
            cuda.download(&cu_out, &mut got_cuda).unwrap();
            assert_close_slice(&got_cuda, leto_out.storage().as_slice(), 0.0, 1e-5);

            let t_cuda = Instant::now();
            for _ in 0..ITERS {
                hephaestus_cuda::kron_into(
                    cuda,
                    CudaStridedOperand {
                        buffer: &cu_a,
                        layout: &layout_a,
                    },
                    CudaStridedOperand {
                        buffer: &cu_b,
                        layout: &layout_b,
                    },
                    CudaStridedOperand {
                        buffer: &cu_out,
                        layout: &out_layout,
                    },
                )
                .unwrap();
            }
            wait_cuda(cuda);
            println!(
                "GPU (CUDA):   {} ns/iter",
                elapsed_per_iter(t_cuda.elapsed()).as_nanos()
            );
        }

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out =
                leto_ops::kron(black_box(&leto_a.view()), black_box(&leto_b.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        let t_na = Instant::now();
        for _ in 0..ITERS {
            let _out = nalgebra_kron(black_box(&na_a), black_box(&na_b));
        }
        println!(
            "CPU (nalgebra):{} ns/iter\n",
            elapsed_per_iter(t_na.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 9. Dot Product (f32, size 65536)
    // ────────────────────────────────────────────────────────────────────────
    {
        println!("--- Benchmarking: Dot Product (f32, N={LINALG_LEN}) ---");
        let host_a: Vec<f32> = (0..LINALG_LEN)
            .map(|i| (i as f32 * 0.17 + 1.0) * 1e-4)
            .collect();
        let host_b: Vec<f32> = (0..LINALG_LEN)
            .map(|i| (i as f32 * 0.31 + 2.0) * 1e-4)
            .collect();

        // Leto
        let leto_a = leto::Array::from_shape_vec([LINALG_LEN], host_a.clone()).unwrap();
        let leto_b = leto::Array::from_shape_vec([LINALG_LEN], host_b.clone()).unwrap();
        let leto_out = leto_ops::dot(&leto_a.view(), &leto_b.view()).unwrap();
        let nd_a = NdArray1::from_vec(host_a.clone());
        let nd_b = NdArray1::from_vec(host_b.clone());

        let layout1d = leto::Layout::c_contiguous([LINALG_LEN]).unwrap();

        // WGPU
        let wg_a = wgpu_dev.upload(&host_a).unwrap();
        let wg_b = wgpu_dev.upload(&host_b).unwrap();
        let wg_out = dot(
            &wgpu_dev,
            WgpuStridedOperand {
                buffer: &wg_a,
                layout: &layout1d,
            },
            WgpuStridedOperand {
                buffer: &wg_b,
                layout: &layout1d,
            },
        )
        .unwrap();
        wait_wgpu(&wgpu_dev);
        let mut got_wgpu = [0.0f32; 1];
        wgpu_dev.download(&wg_out, &mut got_wgpu).unwrap();
        assert!((got_wgpu[0] - leto_out).abs() < 1e-2 * leto_out.abs());

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            let _out = dot(
                &wgpu_dev,
                WgpuStridedOperand {
                    buffer: &wg_a,
                    layout: &layout1d,
                },
                WgpuStridedOperand {
                    buffer: &wg_b,
                    layout: &layout1d,
                },
            )
            .unwrap();
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        // CUDA
        if let Some(ref cuda) = cuda_dev {
            let cu_a = cuda.upload(&host_a).unwrap();
            let cu_b = cuda.upload(&host_b).unwrap();
            let cu_out = hephaestus_cuda::dot(
                cuda,
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
            wait_cuda(cuda);
            let mut got_cuda = [0.0f32; 1];
            cuda.download(&cu_out, &mut got_cuda).unwrap();
            assert!((got_cuda[0] - leto_out).abs() < 1e-2 * leto_out.abs());

            let t_cuda = Instant::now();
            for _ in 0..ITERS {
                let _out = hephaestus_cuda::dot(
                    cuda,
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
            }
            wait_cuda(cuda);
            println!(
                "GPU (CUDA):   {} ns/iter",
                elapsed_per_iter(t_cuda.elapsed()).as_nanos()
            );
        }

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::dot(black_box(&leto_a.view()), black_box(&leto_b.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        let t_nd = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(&nd_a).dot(black_box(&nd_b));
        }
        println!(
            "CPU (ndarray):{} ns/iter\n",
            elapsed_per_iter(t_nd.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 10. Trace (f32, 256x256)
    // ────────────────────────────────────────────────────────────────────────
    {
        let n = 256usize;
        println!("--- Benchmarking: Trace (f32, {n}x{n}) ---");
        let host_m: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.17 + 1.0) * 1e-3).collect();

        // Leto
        let leto_m = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_out = leto_ops::trace(&leto_m.view()).unwrap();
        let nd_m = NdArray2::from_shape_vec([n, n], host_m.clone()).unwrap();

        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();

        // WGPU
        let wg_m = wgpu_dev.upload(&host_m).unwrap();
        let wg_out = trace(
            &wgpu_dev,
            WgpuStridedOperand {
                buffer: &wg_m,
                layout: &layout2d,
            },
        )
        .unwrap();
        wait_wgpu(&wgpu_dev);
        let mut got_wgpu = [0.0f32; 1];
        wgpu_dev.download(&wg_out, &mut got_wgpu).unwrap();
        assert!((got_wgpu[0] - leto_out).abs() < 1e-2 * leto_out.abs());

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            let _out = trace(
                &wgpu_dev,
                WgpuStridedOperand {
                    buffer: &wg_m,
                    layout: &layout2d,
                },
            )
            .unwrap();
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        // CUDA
        if let Some(ref cuda) = cuda_dev {
            let cu_m = cuda.upload(&host_m).unwrap();
            let cu_out = hephaestus_cuda::trace(
                cuda,
                CudaStridedOperand {
                    buffer: &cu_m,
                    layout: &layout2d,
                },
            )
            .unwrap();
            wait_cuda(cuda);
            let mut got_cuda = [0.0f32; 1];
            cuda.download(&cu_out, &mut got_cuda).unwrap();
            assert!((got_cuda[0] - leto_out).abs() < 1e-2 * leto_out.abs());

            let t_cuda = Instant::now();
            for _ in 0..ITERS {
                let _out = hephaestus_cuda::trace(
                    cuda,
                    CudaStridedOperand {
                        buffer: &cu_m,
                        layout: &layout2d,
                    },
                )
                .unwrap();
            }
            wait_cuda(cuda);
            println!(
                "GPU (CUDA):   {} ns/iter",
                elapsed_per_iter(t_cuda.elapsed()).as_nanos()
            );
        }

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::trace(black_box(&leto_m.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        let t_nd = Instant::now();
        for _ in 0..ITERS {
            let mut trace_val = 0.0f32;
            for i in 0..n {
                trace_val += black_box(&nd_m)[(i, i)];
            }
            black_box(trace_val);
        }
        println!(
            "CPU (ndarray):{} ns/iter\n",
            elapsed_per_iter(t_nd.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 11. Matrix Rank (f32, 64x64 diagonal rank 32)
    // ────────────────────────────────────────────────────────────────────────
    {
        let n = 64usize;
        println!("--- Benchmarking: Matrix Rank (f32, {n}x{n} diagonal rank 32) ---");
        let mut host_m = vec![0.0f32; n * n];
        for i in 0..32 {
            host_m[i * n + i] = 1.5f32;
        }

        // Leto
        let leto_m = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_out = leto_ops::matrix_rank(&leto_m.view()).unwrap();

        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();

        // WGPU
        let wg_m = wgpu_dev.upload(&host_m).unwrap();
        let rank_wgpu = matrix_rank(
            &wgpu_dev,
            WgpuStridedOperand {
                buffer: &wg_m,
                layout: &layout2d,
            },
        )
        .unwrap();
        assert_eq!(rank_wgpu, leto_out);

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            let _out = matrix_rank(
                &wgpu_dev,
                WgpuStridedOperand {
                    buffer: &wg_m,
                    layout: &layout2d,
                },
            )
            .unwrap();
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        // CUDA
        if let Some(ref cuda) = cuda_dev {
            let cu_m = cuda.upload(&host_m).unwrap();
            let rank_cuda = hephaestus_cuda::matrix_rank(
                cuda,
                CudaStridedOperand {
                    buffer: &cu_m,
                    layout: &layout2d,
                },
            )
            .unwrap();
            assert_eq!(rank_cuda, leto_out);

            let t_cuda = Instant::now();
            for _ in 0..ITERS {
                let _out = hephaestus_cuda::matrix_rank(
                    cuda,
                    CudaStridedOperand {
                        buffer: &cu_m,
                        layout: &layout2d,
                    },
                )
                .unwrap();
            }
            wait_cuda(cuda);
            println!(
                "GPU (CUDA):   {} ns/iter",
                elapsed_per_iter(t_cuda.elapsed()).as_nanos()
            );
        }

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::matrix_rank(black_box(&leto_m.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 12. Determinant (f32, 64x64 diagonal)
    // ────────────────────────────────────────────────────────────────────────
    {
        let n = 64usize;
        println!("--- Benchmarking: Determinant (f32, {n}x{n} diagonal) ---");
        let mut host_m = vec![0.0f32; n * n];
        for i in 0..n {
            host_m[i * n + i] = 1.0 + (i as f32 * 1.0e-3);
        }

        let leto_m = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_out = leto_ops::det(&leto_m.view()).unwrap();
        let nd_m = NdArray2::from_shape_vec([n, n], host_m.clone()).unwrap();
        let nd_out = (0..n).map(|i| nd_m[(i, i)]).product::<f32>();
        let na_m = DMatrix::from_row_slice(n, n, &host_m);
        let na_out = na_m.determinant();

        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();
        let wg_m = wgpu_dev.upload(&host_m).unwrap();
        let wg_out = det(
            &wgpu_dev,
            WgpuStridedOperand {
                buffer: &wg_m,
                layout: &layout2d,
            },
        )
        .unwrap();
        wait_wgpu(&wgpu_dev);
        let mut got_wgpu = [0.0f32; 1];
        wgpu_dev.download(&wg_out, &mut got_wgpu).unwrap();
        assert!((got_wgpu[0] - leto_out).abs() <= 1.0e-4 * leto_out.abs().max(1.0));
        assert!((got_wgpu[0] - nd_out).abs() <= 1.0e-4 * nd_out.abs().max(1.0));
        assert!((got_wgpu[0] - na_out).abs() <= 1.0e-4 * na_out.abs().max(1.0));

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            let _out = det(
                &wgpu_dev,
                WgpuStridedOperand {
                    buffer: &wg_m,
                    layout: &layout2d,
                },
            )
            .unwrap();
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        // CUDA
        if let Some(ref cuda) = cuda_dev {
            let cu_m = cuda.upload(&host_m).unwrap();
            let cu_out = hephaestus_cuda::det(
                cuda,
                CudaStridedOperand {
                    buffer: &cu_m,
                    layout: &layout2d,
                },
            )
            .unwrap();
            wait_cuda(cuda);
            let mut got_cuda = [0.0f32; 1];
            cuda.download(&cu_out, &mut got_cuda).unwrap();
            assert!((got_cuda[0] - leto_out).abs() <= 1.0e-4 * leto_out.abs().max(1.0));

            let t_cuda = Instant::now();
            for _ in 0..ITERS {
                let _out = hephaestus_cuda::det(
                    cuda,
                    CudaStridedOperand {
                        buffer: &cu_m,
                        layout: &layout2d,
                    },
                )
                .unwrap();
            }
            wait_cuda(cuda);
            println!(
                "GPU (CUDA):   {} ns/iter",
                elapsed_per_iter(t_cuda.elapsed()).as_nanos()
            );
        }

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::det(black_box(&leto_m.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        let t_nd = Instant::now();
        for _ in 0..ITERS {
            let _out = (0..n).map(|i| black_box(&nd_m)[(i, i)]).product::<f32>();
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_nd.elapsed()).as_nanos()
        );

        let t_na = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(&na_m).determinant();
        }
        println!(
            "CPU (nalgebra):{} ns/iter\n",
            elapsed_per_iter(t_na.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 13. Dense Decompositions (f32)
    // ────────────────────────────────────────────────────────────────────────
    {
        let n = 32usize;
        println!("--- Benchmarking: Cholesky Decomposition (f32, {n}x{n} SPD) ---");
        let mut host_m = vec![0.0f32; n * n];
        for row in 0..n {
            for col in 0..n {
                host_m[row * n + col] = if row == col {
                    (n as f32) + 2.0
                } else {
                    1.0 / (1.0 + row.abs_diff(col) as f32)
                };
            }
        }
        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();
        let leto_m = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_out = leto_ops::cholesky_decompose(&leto_m.view()).unwrap();
        let na_m = DMatrix::from_row_slice(n, n, &host_m);
        let na_out = na_m.clone().cholesky().expect("invariant: benchmark SPD");

        let wg_m = wgpu_dev.upload(&host_m).unwrap();
        let wg_out = cholesky_decompose(
            &wgpu_dev,
            WgpuStridedOperand {
                buffer: &wg_m,
                layout: &layout2d,
            },
        )
        .unwrap();
        wait_wgpu(&wgpu_dev);
        let mut got_lower = vec![0.0f32; n * n];
        wgpu_dev.download(wg_out.lower(), &mut got_lower).unwrap();
        assert_close_slice(
            &got_lower,
            leto::Storage::as_slice(leto_out.lower().storage()),
            0.0,
            1.0e-5,
        );
        let na_lower = na_out.l();
        let mut na_lower_row_major = Vec::with_capacity(n * n);
        for row in 0..n {
            for col in 0..n {
                na_lower_row_major.push(na_lower[(row, col)]);
            }
        }
        assert_close_slice(&got_lower, &na_lower_row_major, 1.0e-4, 1.0e-5);

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            let _out = cholesky_decompose(
                &wgpu_dev,
                WgpuStridedOperand {
                    buffer: &wg_m,
                    layout: &layout2d,
                },
            )
            .unwrap();
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::cholesky_decompose(black_box(&leto_m.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        let t_na = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(&na_m)
                .clone()
                .cholesky()
                .expect("invariant: benchmark SPD");
        }
        println!(
            "CPU (nalgebra):{} ns/iter\n",
            elapsed_per_iter(t_na.elapsed()).as_nanos()
        );
    }

    {
        let n = 32usize;
        println!("--- Benchmarking: LU Decomposition (f32, {n}x{n}) ---");
        let host_m: Vec<f32> = (0..(n * n))
            .map(|idx| {
                let row = idx / n;
                let col = idx % n;
                if row == col {
                    4.0 + row as f32 * 0.01
                } else {
                    ((row + 2 * col + 1) % 7) as f32 * 0.01
                }
            })
            .collect();
        let layout2d = leto::Layout::c_contiguous([n, n]).unwrap();
        let leto_m = leto::Array::from_shape_vec([n, n], host_m.clone()).unwrap();
        let leto_out = leto_ops::lu_decompose(&leto_m.view()).unwrap();
        let na_m = DMatrix::from_row_slice(n, n, &host_m);
        let na_det = na_m.clone().lu().determinant();

        let wg_m = wgpu_dev.upload(&host_m).unwrap();
        let wg_out = lu_decompose(
            &wgpu_dev,
            WgpuStridedOperand {
                buffer: &wg_m,
                layout: &layout2d,
            },
        )
        .unwrap();
        wait_wgpu(&wgpu_dev);
        let mut got_factors = vec![0.0f32; n * n];
        wgpu_dev
            .download(wg_out.factors(), &mut got_factors)
            .unwrap();
        assert_close_slice(
            &got_factors,
            leto::Storage::as_slice(leto_out.factors().storage()),
            0.0,
            1.0e-5,
        );
        assert!((wg_out.det() - na_det).abs() <= 1.0e-4 * na_det.abs().max(1.0));

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            let _out = lu_decompose(
                &wgpu_dev,
                WgpuStridedOperand {
                    buffer: &wg_m,
                    layout: &layout2d,
                },
            )
            .unwrap();
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::lu_decompose(black_box(&leto_m.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        let t_na = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(&na_m).clone().lu();
        }
        println!(
            "CPU (nalgebra):{} ns/iter\n",
            elapsed_per_iter(t_na.elapsed()).as_nanos()
        );
    }

    {
        let rows = 48usize;
        let cols = 24usize;
        println!("--- Benchmarking: QR Decomposition (f32, {rows}x{cols}) ---");
        let host_m: Vec<f32> = (0..(rows * cols))
            .map(|idx| {
                let row = idx / cols;
                let col = idx % cols;
                if row == col {
                    3.0
                } else {
                    ((row * 3 + col + 1) % 11) as f32 * 0.01
                }
            })
            .collect();
        let layout2d = leto::Layout::c_contiguous([rows, cols]).unwrap();
        let leto_m = leto::Array::from_shape_vec([rows, cols], host_m.clone()).unwrap();
        let leto_out = leto_ops::qr_decompose(&leto_m.view()).unwrap();
        let na_m = DMatrix::from_row_slice(rows, cols, &host_m);

        let wg_m = wgpu_dev.upload(&host_m).unwrap();
        let wg_out = qr_decompose(
            &wgpu_dev,
            WgpuStridedOperand {
                buffer: &wg_m,
                layout: &layout2d,
            },
        )
        .unwrap();
        wait_wgpu(&wgpu_dev);
        let mut got_r = vec![0.0f32; rows * cols];
        wgpu_dev.download(wg_out.r_buffer(), &mut got_r).unwrap();
        let leto_r = leto_out.r();
        assert_close_slice(
            &got_r,
            leto::Storage::as_slice(leto_r.storage()),
            0.0,
            1.0e-5,
        );

        let t_wgpu = Instant::now();
        for _ in 0..ITERS {
            let _out = qr_decompose(
                &wgpu_dev,
                WgpuStridedOperand {
                    buffer: &wg_m,
                    layout: &layout2d,
                },
            )
            .unwrap();
        }
        wait_wgpu(&wgpu_dev);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
        );

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let _out = leto_ops::qr_decompose(black_box(&leto_m.view())).unwrap();
        }
        println!(
            "CPU (Leto):   {} ns/iter",
            elapsed_per_iter(t_leto.elapsed()).as_nanos()
        );

        let t_na = Instant::now();
        for _ in 0..ITERS {
            let _out = black_box(&na_m).clone().qr();
        }
        println!(
            "CPU (nalgebra):{} ns/iter\n",
            elapsed_per_iter(t_na.elapsed()).as_nanos()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 14. Norms (L1, L2, Max) (f32, N=65536)
    // ────────────────────────────────────────────────────────────────────────
    {
        let ops = vec![("L1", 0), ("L2", 1), ("Max", 2)];

        for (name, op_idx) in ops {
            println!("--- Benchmarking: Norm {name} (f32, N={LINALG_LEN}) ---");
            let host_a: Vec<f32> = (0..LINALG_LEN)
                .map(|i| (i as f32 * 0.17 - 3.0) * 1e-4)
                .collect();

            // Leto
            let leto_a = leto::Array::from_shape_vec([LINALG_LEN], host_a.clone()).unwrap();
            let leto_out = match op_idx {
                0 => leto_ops::norm_l1(&leto_a.view()).unwrap(),
                1 => leto_ops::norm_l2(&leto_a.view()).unwrap(),
                _ => leto_ops::norm_max(&leto_a.view()).unwrap(),
            };

            let layout1d = leto::Layout::c_contiguous([LINALG_LEN]).unwrap();

            // WGPU
            let wg_a = wgpu_dev.upload(&host_a).unwrap();
            let wg_out = match op_idx {
                0 => norm_l1(
                    &wgpu_dev,
                    WgpuStridedOperand {
                        buffer: &wg_a,
                        layout: &layout1d,
                    },
                )
                .unwrap(),
                1 => norm_l2(
                    &wgpu_dev,
                    WgpuStridedOperand {
                        buffer: &wg_a,
                        layout: &layout1d,
                    },
                )
                .unwrap(),
                _ => norm_max(
                    &wgpu_dev,
                    WgpuStridedOperand {
                        buffer: &wg_a,
                        layout: &layout1d,
                    },
                )
                .unwrap(),
            };
            wait_wgpu(&wgpu_dev);
            let mut got_wgpu = [0.0f32; 1];
            wgpu_dev.download(&wg_out, &mut got_wgpu).unwrap();
            assert!((got_wgpu[0] - leto_out).abs() < 1e-2 * leto_out.abs().max(1e-5));

            let t_wgpu = Instant::now();
            for _ in 0..ITERS {
                let _out = match op_idx {
                    0 => norm_l1(
                        &wgpu_dev,
                        WgpuStridedOperand {
                            buffer: &wg_a,
                            layout: &layout1d,
                        },
                    )
                    .unwrap(),
                    1 => norm_l2(
                        &wgpu_dev,
                        WgpuStridedOperand {
                            buffer: &wg_a,
                            layout: &layout1d,
                        },
                    )
                    .unwrap(),
                    _ => norm_max(
                        &wgpu_dev,
                        WgpuStridedOperand {
                            buffer: &wg_a,
                            layout: &layout1d,
                        },
                    )
                    .unwrap(),
                };
            }
            wait_wgpu(&wgpu_dev);
            println!(
                "GPU (WGPU):   {} ns/iter",
                elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
            );

            // CUDA
            if let Some(ref cuda) = cuda_dev {
                let cu_a = cuda.upload(&host_a).unwrap();
                let cu_out = match op_idx {
                    0 => hephaestus_cuda::norm_l1(
                        cuda,
                        CudaStridedOperand {
                            buffer: &cu_a,
                            layout: &layout1d,
                        },
                    )
                    .unwrap(),
                    1 => hephaestus_cuda::norm_l2(
                        cuda,
                        CudaStridedOperand {
                            buffer: &cu_a,
                            layout: &layout1d,
                        },
                    )
                    .unwrap(),
                    _ => hephaestus_cuda::norm_max(
                        cuda,
                        CudaStridedOperand {
                            buffer: &cu_a,
                            layout: &layout1d,
                        },
                    )
                    .unwrap(),
                };
                wait_cuda(cuda);
                let mut got_cuda = [0.0f32; 1];
                cuda.download(&cu_out, &mut got_cuda).unwrap();
                assert!((got_cuda[0] - leto_out).abs() < 1e-2 * leto_out.abs().max(1e-5));

                let t_cuda = Instant::now();
                for _ in 0..ITERS {
                    let _out = match op_idx {
                        0 => hephaestus_cuda::norm_l1(
                            cuda,
                            CudaStridedOperand {
                                buffer: &cu_a,
                                layout: &layout1d,
                            },
                        )
                        .unwrap(),
                        1 => hephaestus_cuda::norm_l2(
                            cuda,
                            CudaStridedOperand {
                                buffer: &cu_a,
                                layout: &layout1d,
                            },
                        )
                        .unwrap(),
                        _ => hephaestus_cuda::norm_max(
                            cuda,
                            CudaStridedOperand {
                                buffer: &cu_a,
                                layout: &layout1d,
                            },
                        )
                        .unwrap(),
                    };
                }
                wait_cuda(cuda);
                println!(
                    "GPU (CUDA):   {} ns/iter",
                    elapsed_per_iter(t_cuda.elapsed()).as_nanos()
                );
            }

            let t_leto = Instant::now();
            for _ in 0..ITERS {
                let _out = match op_idx {
                    0 => leto_ops::norm_l1(black_box(&leto_a.view())).unwrap(),
                    1 => leto_ops::norm_l2(black_box(&leto_a.view())).unwrap(),
                    _ => leto_ops::norm_max(black_box(&leto_a.view())).unwrap(),
                };
            }
            println!(
                "CPU (Leto):   {} ns/iter\n",
                elapsed_per_iter(t_leto.elapsed()).as_nanos()
            );
        }
    }
}
