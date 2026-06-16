//! Comparative benchmark of Hephaestus GPU (WGPU) vs CPU (Leto & ndarray).
//!
//! Benchmarks:
//! 1. Contiguous Add (f32, size 1<<20)
//! 2. Contiguous Exp (f32, size 1<<20)
//! 3. Contiguous Sum (f32, size 1<<20)
//! 4. Axis Reductions (`sum_axis_into`/`min_axis_into`/`max_axis_into`/`mean_axis_into`, 256x256)
//! 5. Matrix Multiplication (`matmul` 64x64 and 256x256)
//! 6. Cumulative Sum (`cumsum_axis_into`, 256x256 over axis 1)
//! 7. Matrix Power (`matpow`, 64x64 exponent 5)
//! 8. Kronecker Product (`kron`, 64x64 ⊗ 8x8)
//! 9. Vector Dot Product (`dot`, size 65,536)
//! 10. Matrix Trace (`trace`, size 256x256)
//! 11. Norms (`norm_l1`, `norm_l2`, `norm_max`, size 65,536)
//!
//! Validates results across all three backends to ensure correctness.

use std::hint::black_box;
use std::time::{Duration, Instant};

use hephaestus_core::BlockWidth;
use hephaestus_wgpu::{
    binary_elementwise_into, cumsum_axis_into, dot, kron, matmul, matpow, max_axis_into,
    mean_axis_into, min_axis_into, norm_l1, norm_l2, norm_max, reduction, sum_axis_into, trace,
    unary_elementwise_into, AddOp, ComputeDevice, ExpOp, StridedOperand, SumOp, WgpuDevice,
};
use leto::Storage;
use nalgebra::{DMatrix, DVector};
use ndarray::Array2 as NdArray2;
use ndarray::{Array1 as NdArray1, Axis};

const LEN: usize = 1 << 20; // 1,048,576 elements for elementwise
const LINALG_LEN: usize = 1 << 16; // 65,536 elements for dot/norms
const ITERS: usize = 50;

fn wait(device: &WgpuDevice) {
    device
        .inner()
        .poll(wgpu::PollType::Wait)
        .expect("invariant: benchmark device poll succeeds");
}

fn elapsed_per_iter(elapsed: Duration) -> Duration {
    elapsed / u32::try_from(ITERS).expect("invariant: iteration count fits u32")
}

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

    // 1. Setup GPU Device
    let device = match WgpuDevice::try_default("hephaestus-comparative-bench") {
        Ok(device) => device,
        Err(e) => {
            eprintln!("skipping benchmark: WGPU device could not be acquired: {e}");
            return;
        }
    };
    println!("GPU Backend: {}\n", device.backend_name());

    // 2. Prepare pinned host inputs (reproducible deterministic values)
    let host_a: Vec<f32> = (0..LEN).map(|i| (i as f32 * 0.731 + 1.0) * 1e-7).collect();
    let host_b: Vec<f32> = (0..LEN).map(|i| (i as f32 * 0.317 + 2.0) * 1e-7).collect();

    // ─── Verification & Validation: Elementwise & Reductions ───
    let leto_a = leto::Array::from_shape_vec([LEN], host_a.clone()).unwrap();
    let leto_b = leto::Array::from_shape_vec([LEN], host_b.clone()).unwrap();

    let mut leto_add_out = leto::Array::zeros([LEN]);
    leto_ops::add(&leto_a.view(), &leto_b.view(), &mut leto_add_out.view_mut()).unwrap();
    let leto_exp_out = leto_ops::unary_map(leto_ops::ExpOp, &leto_a.view()).unwrap();
    let leto_sum_out = leto_ops::sum(&leto_a.view());

    let nd_a = NdArray1::from_vec(host_a.clone());
    let nd_b = NdArray1::from_vec(host_b.clone());
    let nd_add_out = &nd_a + &nd_b;
    let nd_exp_out = nd_a.mapv(f32::exp);
    let nd_sum_out = nd_a.sum();

    let gpu_a = device.upload(&host_a).expect("upload a");
    let gpu_b = device.upload(&host_b).expect("upload b");
    let gpu_add_out = device.alloc_zeroed::<f32>(LEN).expect("alloc gpu_add_out");
    let gpu_exp_out = device.alloc_zeroed::<f32>(LEN).expect("alloc gpu_exp_out");

    binary_elementwise_into::<AddOp, f32>(
        &device,
        &gpu_a,
        &gpu_b,
        &gpu_add_out,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    unary_elementwise_into::<ExpOp, f32>(&device, &gpu_a, &gpu_exp_out, BlockWidth::DEFAULT)
        .unwrap();
    let gpu_sum_buf = reduction::<SumOp, f32>(&device, &gpu_a).unwrap();
    wait(&device);

    let mut host_add_verify = vec![0.0f32; LEN];
    let mut host_exp_verify = vec![0.0f32; LEN];
    let mut host_sum_verify = [0.0f32; 1];

    device.download(&gpu_add_out, &mut host_add_verify).unwrap();
    device.download(&gpu_exp_out, &mut host_exp_verify).unwrap();
    device.download(&gpu_sum_buf, &mut host_sum_verify).unwrap();

    let leto_add_slice = leto_add_out.storage().as_slice();
    let nd_add_slice = nd_add_out.as_slice().unwrap();
    for i in 0..LEN {
        assert!((host_add_verify[i] - leto_add_slice[i]).abs() < 1e-5);
        assert!((host_add_verify[i] - nd_add_slice[i]).abs() < 1e-5);
    }
    let leto_exp_slice = leto_exp_out.storage().as_slice();
    let nd_exp_slice = nd_exp_out.as_slice().unwrap();
    for i in 0..LEN {
        assert!((host_exp_verify[i] - leto_exp_slice[i]).abs() < 1e-5);
        assert!((host_exp_verify[i] - nd_exp_slice[i]).abs() < 1e-5);
    }
    assert!((host_sum_verify[0] - leto_sum_out).abs() < 1e-2 * leto_sum_out.abs());
    assert!((host_sum_verify[0] - nd_sum_out).abs() < 1e-2 * nd_sum_out.abs());

    // ─── Benchmarking Core Ops ───
    println!("--- Benchmarking: Elementwise Add (f32, N={LEN}) ---");
    let t_gpu_add = Instant::now();
    for _ in 0..ITERS {
        binary_elementwise_into::<AddOp, f32>(
            &device,
            &gpu_a,
            &gpu_b,
            &gpu_add_out,
            BlockWidth::DEFAULT,
        )
        .unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu_add.elapsed()).as_nanos()
    );

    let t_leto_add = Instant::now();
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
        elapsed_per_iter(t_leto_add.elapsed()).as_nanos()
    );

    let t_nd_add = Instant::now();
    for _ in 0..ITERS {
        let _c = black_box(&nd_a) + black_box(&nd_b);
    }
    println!(
        "CPU (ndarray):{} ns/iter\n",
        elapsed_per_iter(t_nd_add.elapsed()).as_nanos()
    );

    println!("--- Benchmarking: Elementwise Exp (f32, N={LEN}) ---");
    let t_gpu_exp = Instant::now();
    for _ in 0..ITERS {
        unary_elementwise_into::<ExpOp, f32>(&device, &gpu_a, &gpu_exp_out, BlockWidth::DEFAULT)
            .unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu_exp.elapsed()).as_nanos()
    );

    let t_leto_exp = Instant::now();
    for _ in 0..ITERS {
        let _out = leto_ops::unary_map(leto_ops::ExpOp, black_box(&leto_a.view())).unwrap();
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto_exp.elapsed()).as_nanos()
    );

    let t_nd_exp = Instant::now();
    for _ in 0..ITERS {
        let _c = black_box(&nd_a).mapv(f32::exp);
    }
    println!(
        "CPU (ndarray):{} ns/iter\n",
        elapsed_per_iter(t_nd_exp.elapsed()).as_nanos()
    );

    println!("--- Benchmarking: Sum Reduction (f32, N={LEN}) ---");
    let t_gpu_sum = Instant::now();
    for _ in 0..ITERS {
        let _res = reduction::<SumOp, f32>(&device, &gpu_a).unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu_sum.elapsed()).as_nanos()
    );

    let t_leto_sum = Instant::now();
    for _ in 0..ITERS {
        let _val = leto_ops::sum(black_box(&leto_a.view()));
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto_sum.elapsed()).as_nanos()
    );

    let t_nd_sum = Instant::now();
    for _ in 0..ITERS {
        let _val = black_box(&nd_a).sum();
    }
    println!(
        "CPU (ndarray):{} ns/iter\n",
        elapsed_per_iter(t_nd_sum.elapsed()).as_nanos()
    );

    // ─── 4. Axis Sum Benchmark ───
    let axis_n = 256usize;
    let host_axis: Vec<f32> = (0..axis_n * axis_n)
        .map(|i| (i as f32 * 0.43 + 1.0) * 1e-4)
        .collect();
    let leto_axis = leto::Array::from_shape_vec([axis_n, axis_n], host_axis.clone()).unwrap();
    let leto_axis_out = leto_ops::sum_axis(&leto_axis.view(), 0).unwrap();
    let leto_axis_min = leto_ops::min_axis(&leto_axis.view(), 0).unwrap();
    let leto_axis_max = leto_ops::max_axis(&leto_axis.view(), 0).unwrap();
    let leto_axis_mean = leto_ops::mean_axis(&leto_axis.view(), 0).unwrap();
    let nd_axis = NdArray2::from_shape_vec([axis_n, axis_n], host_axis.clone()).unwrap();
    let nd_axis_out = nd_axis.sum_axis(Axis(0));
    let nd_axis_min = nd_axis.fold_axis(Axis(0), f32::INFINITY, |acc, x| acc.min(*x));
    let nd_axis_max = nd_axis.fold_axis(Axis(0), f32::NEG_INFINITY, |acc, x| acc.max(*x));
    let nd_axis_mean = nd_axis
        .mean_axis(Axis(0))
        .expect("invariant: benchmark axis is non-empty");
    let na_axis = DMatrix::from_row_slice(axis_n, axis_n, &host_axis);
    let na_axis_out = nalgebra_sum_axis(&na_axis, 0);
    let na_axis_min = nalgebra_min_axis(&na_axis, 0);
    let na_axis_max = nalgebra_max_axis(&na_axis, 0);
    let na_axis_mean = nalgebra_mean_axis(&na_axis, 0);

    let gpu_axis = device.upload(&host_axis).unwrap();
    let gpu_axis_out = device.alloc_zeroed::<f32>(axis_n).unwrap();
    let gpu_axis_min = device.alloc_zeroed::<f32>(axis_n).unwrap();
    let gpu_axis_max = device.alloc_zeroed::<f32>(axis_n).unwrap();
    let gpu_axis_mean = device.alloc_zeroed::<f32>(axis_n).unwrap();
    let axis_input_layout = leto::Layout::c_contiguous([axis_n, axis_n]).unwrap();
    let axis_output_layout = leto::Layout::c_contiguous([1, axis_n]).unwrap();
    let axis_input = StridedOperand {
        buffer: &gpu_axis,
        layout: &axis_input_layout,
    };
    let axis_output = StridedOperand {
        buffer: &gpu_axis_out,
        layout: &axis_output_layout,
    };
    let axis_min_output = StridedOperand {
        buffer: &gpu_axis_min,
        layout: &axis_output_layout,
    };
    let axis_max_output = StridedOperand {
        buffer: &gpu_axis_max,
        layout: &axis_output_layout,
    };
    let axis_mean_output = StridedOperand {
        buffer: &gpu_axis_mean,
        layout: &axis_output_layout,
    };
    sum_axis_into(&device, axis_input, 0, axis_output, BlockWidth::DEFAULT).unwrap();
    min_axis_into(&device, axis_input, 0, axis_min_output, BlockWidth::DEFAULT).unwrap();
    max_axis_into(&device, axis_input, 0, axis_max_output, BlockWidth::DEFAULT).unwrap();
    mean_axis_into(
        &device,
        axis_input,
        0,
        axis_mean_output,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    wait(&device);

    let mut got_axis = vec![0.0f32; axis_n];
    device.download(&gpu_axis_out, &mut got_axis).unwrap();
    let leto_axis_slice = leto_axis_out.storage().as_slice();
    let nd_axis_slice = nd_axis_out.as_slice().unwrap();
    for i in 0..axis_n {
        let tolerance = 4.0 * f32::EPSILON * leto_axis_slice[i].abs().max(1.0);
        assert!((got_axis[i] - leto_axis_slice[i]).abs() <= tolerance);
        assert!((got_axis[i] - nd_axis_slice[i]).abs() <= tolerance);
        assert!((got_axis[i] - na_axis_out[i]).abs() <= tolerance);
    }
    let mut got_axis_min = vec![0.0f32; axis_n];
    device.download(&gpu_axis_min, &mut got_axis_min).unwrap();
    assert_eq!(got_axis_min, leto_axis_min.storage().as_slice());
    assert_eq!(got_axis_min, nd_axis_min.as_slice().unwrap());
    assert_eq!(got_axis_min, na_axis_min);
    let mut got_axis_max = vec![0.0f32; axis_n];
    device.download(&gpu_axis_max, &mut got_axis_max).unwrap();
    assert_eq!(got_axis_max, leto_axis_max.storage().as_slice());
    assert_eq!(got_axis_max, nd_axis_max.as_slice().unwrap());
    assert_eq!(got_axis_max, na_axis_max);
    let mut got_axis_mean = vec![0.0f32; axis_n];
    device.download(&gpu_axis_mean, &mut got_axis_mean).unwrap();
    let leto_axis_mean_slice = leto_axis_mean.storage().as_slice();
    let nd_axis_mean_slice = nd_axis_mean.as_slice().unwrap();
    for i in 0..axis_n {
        let tolerance = 4.0 * f32::EPSILON * leto_axis_mean_slice[i].abs().max(1.0);
        assert!((got_axis_mean[i] - leto_axis_mean_slice[i]).abs() <= tolerance);
        assert!((got_axis_mean[i] - nd_axis_mean_slice[i]).abs() <= tolerance);
        assert!((got_axis_mean[i] - na_axis_mean[i]).abs() <= tolerance);
    }

    println!("--- Benchmarking: Axis Sum (f32, 256x256 over axis 0) ---");
    let t_gpu_axis = Instant::now();
    for _ in 0..ITERS {
        sum_axis_into(&device, axis_input, 0, axis_output, BlockWidth::DEFAULT).unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu_axis.elapsed()).as_nanos()
    );

    let t_leto_axis = Instant::now();
    for _ in 0..ITERS {
        let _out = leto_ops::sum_axis(black_box(&leto_axis.view()), 0).unwrap();
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto_axis.elapsed()).as_nanos()
    );

    let t_nd_axis = Instant::now();
    for _ in 0..ITERS {
        let _out = black_box(&nd_axis).sum_axis(Axis(0));
    }
    println!(
        "CPU (ndarray):{} ns/iter",
        elapsed_per_iter(t_nd_axis.elapsed()).as_nanos()
    );

    let t_na_axis = Instant::now();
    for _ in 0..ITERS {
        let _out = nalgebra_sum_axis(black_box(&na_axis), 0);
    }
    println!(
        "CPU (nalgebra):{} ns/iter\n",
        elapsed_per_iter(t_na_axis.elapsed()).as_nanos()
    );

    println!("--- Benchmarking: Axis Min (f32, 256x256 over axis 0) ---");
    let t_gpu_axis_min = Instant::now();
    for _ in 0..ITERS {
        min_axis_into(&device, axis_input, 0, axis_min_output, BlockWidth::DEFAULT).unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu_axis_min.elapsed()).as_nanos()
    );

    let t_leto_axis_min = Instant::now();
    for _ in 0..ITERS {
        let _out = leto_ops::min_axis(black_box(&leto_axis.view()), 0).unwrap();
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto_axis_min.elapsed()).as_nanos()
    );

    let t_nd_axis_min = Instant::now();
    for _ in 0..ITERS {
        let _out = black_box(&nd_axis).fold_axis(Axis(0), f32::INFINITY, |acc, x| acc.min(*x));
    }
    println!(
        "CPU (ndarray):{} ns/iter",
        elapsed_per_iter(t_nd_axis_min.elapsed()).as_nanos()
    );

    let t_na_axis_min = Instant::now();
    for _ in 0..ITERS {
        let _out = nalgebra_min_axis(black_box(&na_axis), 0);
    }
    println!(
        "CPU (nalgebra):{} ns/iter\n",
        elapsed_per_iter(t_na_axis_min.elapsed()).as_nanos()
    );

    println!("--- Benchmarking: Axis Max (f32, 256x256 over axis 0) ---");
    let t_gpu_axis_max = Instant::now();
    for _ in 0..ITERS {
        max_axis_into(&device, axis_input, 0, axis_max_output, BlockWidth::DEFAULT).unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu_axis_max.elapsed()).as_nanos()
    );

    let t_leto_axis_max = Instant::now();
    for _ in 0..ITERS {
        let _out = leto_ops::max_axis(black_box(&leto_axis.view()), 0).unwrap();
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto_axis_max.elapsed()).as_nanos()
    );

    let t_nd_axis_max = Instant::now();
    for _ in 0..ITERS {
        let _out = black_box(&nd_axis).fold_axis(Axis(0), f32::NEG_INFINITY, |acc, x| acc.max(*x));
    }
    println!(
        "CPU (ndarray):{} ns/iter",
        elapsed_per_iter(t_nd_axis_max.elapsed()).as_nanos()
    );

    let t_na_axis_max = Instant::now();
    for _ in 0..ITERS {
        let _out = nalgebra_max_axis(black_box(&na_axis), 0);
    }
    println!(
        "CPU (nalgebra):{} ns/iter\n",
        elapsed_per_iter(t_na_axis_max.elapsed()).as_nanos()
    );

    println!("--- Benchmarking: Axis Mean (f32, 256x256 over axis 0) ---");
    let t_gpu_axis_mean = Instant::now();
    for _ in 0..ITERS {
        mean_axis_into(
            &device,
            axis_input,
            0,
            axis_mean_output,
            BlockWidth::DEFAULT,
        )
        .unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu_axis_mean.elapsed()).as_nanos()
    );

    let t_leto_axis_mean = Instant::now();
    for _ in 0..ITERS {
        let _out = leto_ops::mean_axis(black_box(&leto_axis.view()), 0).unwrap();
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto_axis_mean.elapsed()).as_nanos()
    );

    let t_nd_axis_mean = Instant::now();
    for _ in 0..ITERS {
        let _out = black_box(&nd_axis)
            .mean_axis(Axis(0))
            .expect("invariant: benchmark axis is non-empty");
    }
    println!(
        "CPU (ndarray):{} ns/iter",
        elapsed_per_iter(t_nd_axis_mean.elapsed()).as_nanos()
    );

    let t_na_axis_mean = Instant::now();
    for _ in 0..ITERS {
        let _out = nalgebra_mean_axis(black_box(&na_axis), 0);
    }
    println!(
        "CPU (nalgebra):{} ns/iter\n",
        elapsed_per_iter(t_na_axis_mean.elapsed()).as_nanos()
    );

    // ─── 5. Matmul Benchmarks ───
    for &n in &[64usize, 256] {
        let host_m_a: Vec<f32> = (0..n * n)
            .map(|i| (i as f32 * 0.731 + 1.0) * 1e-4)
            .collect();
        let host_m_b: Vec<f32> = (0..n * n)
            .map(|i| (i as f32 * 0.317 + 2.0) * 1e-4)
            .collect();

        let leto_m_a = leto::Array::from_shape_vec([n, n], host_m_a.clone()).unwrap();
        let leto_m_b = leto::Array::from_shape_vec([n, n], host_m_b.clone()).unwrap();
        let mut leto_m_out = leto::Array::zeros([n, n]);
        leto_ops::matmul(
            &leto_m_a.view(),
            &leto_m_b.view(),
            &mut leto_m_out.view_mut(),
        )
        .unwrap();

        let nd_m_a = NdArray2::from_shape_vec([n, n], host_m_a.clone()).unwrap();
        let nd_m_b = NdArray2::from_shape_vec([n, n], host_m_b.clone()).unwrap();
        let nd_m_out = nd_m_a.dot(&nd_m_b);

        let na_m_a = DMatrix::from_row_slice(n, n, &host_m_a);
        let na_m_b = DMatrix::from_row_slice(n, n, &host_m_b);
        let na_m_out = &na_m_a * &na_m_b;

        let gpu_m_a = device.upload(&host_m_a).unwrap();
        let gpu_m_b = device.upload(&host_m_b).unwrap();
        let gpu_m_out = device.alloc_zeroed::<f32>(n * n).unwrap();

        let layout_a = leto::Layout::c_contiguous([n, n]).unwrap();
        let layout_b = leto::Layout::c_contiguous([n, n]).unwrap();
        let layout_out = leto::Layout::c_contiguous([n, n]).unwrap();

        matmul(
            &device,
            StridedOperand {
                buffer: &gpu_m_a,
                layout: &layout_a,
            },
            StridedOperand {
                buffer: &gpu_m_b,
                layout: &layout_b,
            },
            StridedOperand {
                buffer: &gpu_m_out,
                layout: &layout_out,
            },
        )
        .unwrap();
        wait(&device);

        let mut verify = vec![0.0f32; n * n];
        device.download(&gpu_m_out, &mut verify).unwrap();
        let leto_slice = leto_m_out.storage().as_slice();
        let nd_slice = nd_m_out.as_slice().unwrap();
        for r in 0..n {
            for c in 0..n {
                let i = r * n + c;
                assert!((verify[i] - leto_slice[i]).abs() < 1e-2 * leto_slice[i].abs().max(1.0));
                assert!((verify[i] - nd_slice[i]).abs() < 1e-2 * nd_slice[i].abs().max(1.0));
                assert!(
                    (verify[i] - na_m_out[(r, c)]).abs() < 1e-2 * na_m_out[(r, c)].abs().max(1.0)
                );
            }
        }

        println!("--- Benchmarking: Matmul {n}x{n} (f32) ---");
        let t_gpu = Instant::now();
        for _ in 0..ITERS {
            matmul(
                &device,
                StridedOperand {
                    buffer: &gpu_m_a,
                    layout: &layout_a,
                },
                StridedOperand {
                    buffer: &gpu_m_b,
                    layout: &layout_b,
                },
                StridedOperand {
                    buffer: &gpu_m_out,
                    layout: &layout_out,
                },
            )
            .unwrap();
        }
        wait(&device);
        println!(
            "GPU (WGPU):   {} ns/iter",
            elapsed_per_iter(t_gpu.elapsed()).as_nanos()
        );

        let t_leto = Instant::now();
        for _ in 0..ITERS {
            let mut out = black_box(leto::Array::zeros([n, n]));
            leto_ops::matmul(
                black_box(&leto_m_a.view()),
                black_box(&leto_m_b.view()),
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
            let _c = black_box(black_box(&nd_m_a).dot(black_box(&nd_m_b)));
        }
        println!(
            "CPU (ndarray):{} ns/iter",
            elapsed_per_iter(t_nd.elapsed()).as_nanos()
        );

        let t_na = Instant::now();
        for _ in 0..ITERS {
            let _c = black_box(black_box(&na_m_a) * black_box(&na_m_b));
        }
        println!(
            "CPU (nalgebra):{} ns/iter\n",
            elapsed_per_iter(t_na.elapsed()).as_nanos()
        );
    }

    // ─── 6. Cumulative Sum Benchmark ───
    let scan_n = 256usize;
    let host_scan: Vec<f32> = (0..scan_n * scan_n)
        .map(|i| (i as f32 * 0.11 + 1.0) * 1e-5)
        .collect();
    let leto_scan = leto::Array::from_shape_vec([scan_n, scan_n], host_scan.clone()).unwrap();
    let leto_scan_out = leto_ops::cumsum(&leto_scan.view(), 1).unwrap();
    let nd_scan = NdArray2::from_shape_vec([scan_n, scan_n], host_scan.clone()).unwrap();
    let nd_scan_out = ndarray_cumsum_axis(&nd_scan, 1);
    let na_scan = DMatrix::from_row_slice(scan_n, scan_n, &host_scan);
    let na_scan_out = nalgebra_cumsum_axis(&na_scan, 1);

    let gpu_scan = device.upload(&host_scan).unwrap();
    let gpu_scan_out = device.alloc_zeroed::<f32>(scan_n * scan_n).unwrap();
    let scan_layout = leto::Layout::c_contiguous([scan_n, scan_n]).unwrap();
    let scan_input = StridedOperand {
        buffer: &gpu_scan,
        layout: &scan_layout,
    };
    let scan_output = StridedOperand {
        buffer: &gpu_scan_out,
        layout: &scan_layout,
    };
    cumsum_axis_into(&device, scan_input, 1, scan_output, BlockWidth::DEFAULT).unwrap();
    wait(&device);

    let mut got_scan = vec![0.0f32; scan_n * scan_n];
    device.download(&gpu_scan_out, &mut got_scan).unwrap();
    let leto_scan_slice = leto_scan_out.storage().as_slice();
    let nd_scan_slice = nd_scan_out.as_slice().unwrap();
    for i in 0..got_scan.len() {
        let tolerance = 8.0 * f32::EPSILON * leto_scan_slice[i].abs().max(1.0);
        assert!((got_scan[i] - leto_scan_slice[i]).abs() <= tolerance);
        assert!((got_scan[i] - nd_scan_slice[i]).abs() <= tolerance);
        assert!((got_scan[i] - na_scan_out[i]).abs() <= tolerance);
    }

    println!("--- Benchmarking: Cumsum (f32, 256x256 over axis 1) ---");
    let t_gpu_scan = Instant::now();
    for _ in 0..ITERS {
        cumsum_axis_into(&device, scan_input, 1, scan_output, BlockWidth::DEFAULT).unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu_scan.elapsed()).as_nanos()
    );

    let t_leto_scan = Instant::now();
    for _ in 0..ITERS {
        let _out = leto_ops::cumsum(black_box(&leto_scan.view()), 1).unwrap();
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto_scan.elapsed()).as_nanos()
    );

    let t_nd_scan = Instant::now();
    for _ in 0..ITERS {
        let _out = ndarray_cumsum_axis(black_box(&nd_scan), 1);
    }
    println!(
        "CPU (ndarray):{} ns/iter",
        elapsed_per_iter(t_nd_scan.elapsed()).as_nanos()
    );

    let t_na_scan = Instant::now();
    for _ in 0..ITERS {
        let _out = nalgebra_cumsum_axis(black_box(&na_scan), 1);
    }
    println!(
        "CPU (nalgebra):{} ns/iter\n",
        elapsed_per_iter(t_na_scan.elapsed()).as_nanos()
    );

    // ─── 7. Matrix Power Benchmark ───
    let pow_n = 64usize;
    let pow_exponent = 5u32;
    let host_pow: Vec<f32> = (0..pow_n * pow_n)
        .map(|i| {
            let row = i / pow_n;
            let col = i % pow_n;
            if row == col {
                1.0 + (row as f32 + 1.0) * 1e-3
            } else {
                ((row + col + 1) as f32) * 1e-5
            }
        })
        .collect();

    let leto_pow_a = leto::Array::from_shape_vec([pow_n, pow_n], host_pow.clone()).unwrap();
    let leto_pow_out = leto_ops::matpow(&leto_pow_a.view(), pow_exponent).unwrap();

    let nd_pow_a = NdArray2::from_shape_vec([pow_n, pow_n], host_pow.clone()).unwrap();
    let nd_pow_out = ndarray_matpow(&nd_pow_a, pow_exponent);

    let na_pow_a = DMatrix::from_row_slice(pow_n, pow_n, &host_pow);
    let na_pow_out = na_pow_a.pow(pow_exponent);

    let gpu_pow_a = device.upload(&host_pow).unwrap();
    let pow_layout = leto::Layout::c_contiguous([pow_n, pow_n]).unwrap();

    let gpu_pow_buf = matpow(
        &device,
        StridedOperand {
            buffer: &gpu_pow_a,
            layout: &pow_layout,
        },
        pow_exponent,
    )
    .unwrap();
    wait(&device);

    let mut verify_pow = vec![0.0f32; pow_n * pow_n];
    device.download(&gpu_pow_buf, &mut verify_pow).unwrap();
    let leto_pow_slice = leto_pow_out.storage().as_slice();
    let nd_pow_slice = nd_pow_out.as_slice().unwrap();
    for r in 0..pow_n {
        for c in 0..pow_n {
            let i = r * pow_n + c;
            let tolerance = 2.0e-3 * leto_pow_slice[i].abs().max(1.0);
            assert!((verify_pow[i] - leto_pow_slice[i]).abs() <= tolerance);
            assert!((verify_pow[i] - nd_pow_slice[i]).abs() <= tolerance);
            assert!((verify_pow[i] - na_pow_out[(r, c)]).abs() <= tolerance);
        }
    }

    println!("--- Benchmarking: Matrix Power (f32, 64x64 exponent 5) ---");
    let t_gpu_pow = Instant::now();
    for _ in 0..ITERS {
        let _out = matpow(
            &device,
            StridedOperand {
                buffer: &gpu_pow_a,
                layout: &pow_layout,
            },
            pow_exponent,
        )
        .unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu_pow.elapsed()).as_nanos()
    );

    let t_leto_pow = Instant::now();
    for _ in 0..ITERS {
        let _out = leto_ops::matpow(black_box(&leto_pow_a.view()), pow_exponent).unwrap();
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto_pow.elapsed()).as_nanos()
    );

    let t_nd_pow = Instant::now();
    for _ in 0..ITERS {
        let _out = ndarray_matpow(black_box(&nd_pow_a), pow_exponent);
    }
    println!(
        "CPU (ndarray):{} ns/iter",
        elapsed_per_iter(t_nd_pow.elapsed()).as_nanos()
    );

    let t_na_pow = Instant::now();
    for _ in 0..ITERS {
        let _out = black_box(&na_pow_a).pow(pow_exponent);
    }
    println!(
        "CPU (nalgebra):{} ns/iter\n",
        elapsed_per_iter(t_na_pow.elapsed()).as_nanos()
    );

    // ─── 7. Kronecker Product Benchmark ───
    let kron_a_rows = 64usize;
    let kron_a_cols = 64usize;
    let kron_b_rows = 8usize;
    let kron_b_cols = 8usize;
    let kron_rows = kron_a_rows * kron_b_rows;
    let kron_cols = kron_a_cols * kron_b_cols;
    let host_kron_a: Vec<f32> = (0..kron_a_rows * kron_a_cols)
        .map(|i| (i as f32 * 0.19 + 1.0) * 1e-3)
        .collect();
    let host_kron_b: Vec<f32> = (0..kron_b_rows * kron_b_cols)
        .map(|i| (i as f32 * 0.37 + 2.0) * 1e-3)
        .collect();

    let leto_kron_a =
        leto::Array::from_shape_vec([kron_a_rows, kron_a_cols], host_kron_a.clone()).unwrap();
    let leto_kron_b =
        leto::Array::from_shape_vec([kron_b_rows, kron_b_cols], host_kron_b.clone()).unwrap();
    let leto_kron_out = leto_ops::kron(&leto_kron_a.view(), &leto_kron_b.view()).unwrap();

    let nd_kron_a =
        NdArray2::from_shape_vec([kron_a_rows, kron_a_cols], host_kron_a.clone()).unwrap();
    let nd_kron_b =
        NdArray2::from_shape_vec([kron_b_rows, kron_b_cols], host_kron_b.clone()).unwrap();
    let mut nd_kron_out = NdArray2::zeros((kron_rows, kron_cols));
    for i in 0..kron_a_rows {
        for j in 0..kron_a_cols {
            let scale = nd_kron_a[(i, j)];
            let row_base = i * kron_b_rows;
            let col_base = j * kron_b_cols;
            for k in 0..kron_b_rows {
                for l in 0..kron_b_cols {
                    nd_kron_out[(row_base + k, col_base + l)] = scale * nd_kron_b[(k, l)];
                }
            }
        }
    }

    let na_kron_a = DMatrix::from_row_slice(kron_a_rows, kron_a_cols, &host_kron_a);
    let na_kron_b = DMatrix::from_row_slice(kron_b_rows, kron_b_cols, &host_kron_b);
    let na_kron_out = nalgebra_kron(&na_kron_a, &na_kron_b);

    let gpu_kron_a = device.upload(&host_kron_a).unwrap();
    let gpu_kron_b = device.upload(&host_kron_b).unwrap();
    let gpu_kron_out = device.alloc_zeroed::<f32>(kron_rows * kron_cols).unwrap();
    let kron_a_layout = leto::Layout::c_contiguous([kron_a_rows, kron_a_cols]).unwrap();
    let kron_b_layout = leto::Layout::c_contiguous([kron_b_rows, kron_b_cols]).unwrap();
    let kron_out_layout = leto::Layout::c_contiguous([kron_rows, kron_cols]).unwrap();

    kron(
        &device,
        StridedOperand {
            buffer: &gpu_kron_a,
            layout: &kron_a_layout,
        },
        StridedOperand {
            buffer: &gpu_kron_b,
            layout: &kron_b_layout,
        },
        StridedOperand {
            buffer: &gpu_kron_out,
            layout: &kron_out_layout,
        },
    )
    .unwrap();
    wait(&device);

    let mut verify_kron = vec![0.0f32; kron_rows * kron_cols];
    device.download(&gpu_kron_out, &mut verify_kron).unwrap();
    let leto_kron_slice = leto_kron_out.storage().as_slice();
    let nd_kron_slice = nd_kron_out.as_slice().unwrap();
    for r in 0..kron_rows {
        for c in 0..kron_cols {
            let i = r * kron_cols + c;
            assert!((verify_kron[i] - leto_kron_slice[i]).abs() <= f32::EPSILON);
            assert!((verify_kron[i] - nd_kron_slice[i]).abs() <= f32::EPSILON);
            assert!((verify_kron[i] - na_kron_out[(r, c)]).abs() <= f32::EPSILON);
        }
    }

    println!("--- Benchmarking: Kronecker Product (f32, 64x64 ⊗ 8x8) ---");
    let t_gpu = Instant::now();
    for _ in 0..ITERS {
        kron(
            &device,
            StridedOperand {
                buffer: &gpu_kron_a,
                layout: &kron_a_layout,
            },
            StridedOperand {
                buffer: &gpu_kron_b,
                layout: &kron_b_layout,
            },
            StridedOperand {
                buffer: &gpu_kron_out,
                layout: &kron_out_layout,
            },
        )
        .unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu.elapsed()).as_nanos()
    );

    let t_leto = Instant::now();
    for _ in 0..ITERS {
        let _out = leto_ops::kron(
            black_box(&leto_kron_a.view()),
            black_box(&leto_kron_b.view()),
        )
        .unwrap();
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto.elapsed()).as_nanos()
    );

    let t_nd = Instant::now();
    for _ in 0..ITERS {
        let mut out = black_box(NdArray2::zeros((kron_rows, kron_cols)));
        for i in 0..kron_a_rows {
            for j in 0..kron_a_cols {
                let scale = black_box(&nd_kron_a)[(i, j)];
                let row_base = i * kron_b_rows;
                let col_base = j * kron_b_cols;
                for k in 0..kron_b_rows {
                    for l in 0..kron_b_cols {
                        out[(row_base + k, col_base + l)] = scale * black_box(&nd_kron_b)[(k, l)];
                    }
                }
            }
        }
        black_box(out);
    }
    println!(
        "CPU (ndarray):{} ns/iter",
        elapsed_per_iter(t_nd.elapsed()).as_nanos()
    );

    let t_na = Instant::now();
    for _ in 0..ITERS {
        let _out = nalgebra_kron(black_box(&na_kron_a), black_box(&na_kron_b));
    }
    println!(
        "CPU (nalgebra):{} ns/iter\n",
        elapsed_per_iter(t_na.elapsed()).as_nanos()
    );

    // ─── 6. Vector Dot Product Benchmark ───
    let host_v_a: Vec<f32> = (0..LINALG_LEN)
        .map(|i| (i as f32 * 0.731 + 1.0) * 1e-5)
        .collect();
    let host_v_b: Vec<f32> = (0..LINALG_LEN)
        .map(|i| (i as f32 * 0.317 + 2.0) * 1e-5)
        .collect();

    let leto_v_a = leto::Array::from_shape_vec([LINALG_LEN], host_v_a.clone()).unwrap();
    let leto_v_b = leto::Array::from_shape_vec([LINALG_LEN], host_v_b.clone()).unwrap();
    let leto_dot_res = leto_ops::dot(&leto_v_a.view(), &leto_v_b.view()).unwrap();

    let nd_v_a = NdArray1::from_vec(host_v_a.clone());
    let nd_v_b = NdArray1::from_vec(host_v_b.clone());
    let nd_dot_res = nd_v_a.dot(&nd_v_b);

    let na_v_a = DVector::from_column_slice(&host_v_a);
    let na_v_b = DVector::from_column_slice(&host_v_b);
    let na_dot_res = na_v_a.dot(&na_v_b);

    let gpu_v_a = device.upload(&host_v_a).unwrap();
    let gpu_v_b = device.upload(&host_v_b).unwrap();
    let layout_v_a = leto::Layout::c_contiguous([LINALG_LEN]).unwrap();
    let layout_v_b = leto::Layout::c_contiguous([LINALG_LEN]).unwrap();

    let gpu_dot_buf = dot(
        &device,
        StridedOperand {
            buffer: &gpu_v_a,
            layout: &layout_v_a,
        },
        StridedOperand {
            buffer: &gpu_v_b,
            layout: &layout_v_b,
        },
    )
    .unwrap();
    wait(&device);

    let mut got_dot = [0.0f32; 1];
    device.download(&gpu_dot_buf, &mut got_dot).unwrap();
    assert!((got_dot[0] - leto_dot_res).abs() < 1e-3 * leto_dot_res.abs());
    assert!((got_dot[0] - nd_dot_res).abs() < 1e-3 * nd_dot_res.abs());
    assert!((got_dot[0] - na_dot_res).abs() < 1e-3 * na_dot_res.abs());

    println!("--- Benchmarking: Dot Product (f32, N={LINALG_LEN}) ---");
    let t_gpu = Instant::now();
    for _ in 0..ITERS {
        let _res = dot(
            &device,
            StridedOperand {
                buffer: &gpu_v_a,
                layout: &layout_v_a,
            },
            StridedOperand {
                buffer: &gpu_v_b,
                layout: &layout_v_b,
            },
        )
        .unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu.elapsed()).as_nanos()
    );

    let t_leto = Instant::now();
    for _ in 0..ITERS {
        let _val = leto_ops::dot(black_box(&leto_v_a.view()), black_box(&leto_v_b.view())).unwrap();
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto.elapsed()).as_nanos()
    );

    let t_nd = Instant::now();
    for _ in 0..ITERS {
        let _val = black_box(black_box(&nd_v_a).dot(black_box(&nd_v_b)));
    }
    println!(
        "CPU (ndarray):{} ns/iter",
        elapsed_per_iter(t_nd.elapsed()).as_nanos()
    );

    let t_na = Instant::now();
    for _ in 0..ITERS {
        let _val = black_box(black_box(&na_v_a).dot(black_box(&na_v_b)));
    }
    println!(
        "CPU (nalgebra):{} ns/iter\n",
        elapsed_per_iter(t_na.elapsed()).as_nanos()
    );

    // ─── 6. Matrix Trace Benchmark ───
    let n_trace = 256usize;
    let host_tr: Vec<f32> = (0..n_trace * n_trace)
        .map(|i| (i as f32 * 0.11 + 1.0) * 1e-5)
        .collect();
    let leto_tr = leto::Array::from_shape_vec([n_trace, n_trace], host_tr.clone()).unwrap();
    let leto_tr_res = leto_ops::trace(&leto_tr.view()).unwrap();

    let nd_tr = NdArray2::from_shape_vec([n_trace, n_trace], host_tr.clone()).unwrap();
    let nd_tr_res = nd_tr.diag().sum();

    let na_tr = DMatrix::from_row_slice(n_trace, n_trace, &host_tr);
    let na_tr_res = na_tr.trace();

    let gpu_tr = device.upload(&host_tr).unwrap();
    let layout_tr = leto::Layout::c_contiguous([n_trace, n_trace]).unwrap();

    let gpu_tr_buf = trace(
        &device,
        StridedOperand {
            buffer: &gpu_tr,
            layout: &layout_tr,
        },
    )
    .unwrap();
    wait(&device);

    let mut got_tr = [0.0f32; 1];
    device.download(&gpu_tr_buf, &mut got_tr).unwrap();
    assert!((got_tr[0] - leto_tr_res).abs() < 1e-3 * leto_tr_res.abs());
    assert!((got_tr[0] - nd_tr_res).abs() < 1e-3 * nd_tr_res.abs());
    assert!((got_tr[0] - na_tr_res).abs() < 1e-3 * na_tr_res.abs());

    println!("--- Benchmarking: Trace (f32, 256x256) ---");
    let t_gpu = Instant::now();
    for _ in 0..ITERS {
        let _res = trace(
            &device,
            StridedOperand {
                buffer: &gpu_tr,
                layout: &layout_tr,
            },
        )
        .unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu.elapsed()).as_nanos()
    );

    let t_leto = Instant::now();
    for _ in 0..ITERS {
        let _val = leto_ops::trace(black_box(&leto_tr.view())).unwrap();
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto.elapsed()).as_nanos()
    );

    let t_nd = Instant::now();
    for _ in 0..ITERS {
        let _val = black_box(black_box(&nd_tr).diag().sum());
    }
    println!(
        "CPU (ndarray):{} ns/iter",
        elapsed_per_iter(t_nd.elapsed()).as_nanos()
    );

    let t_na = Instant::now();
    for _ in 0..ITERS {
        let _val = black_box(black_box(&na_tr).trace());
    }
    println!(
        "CPU (nalgebra):{} ns/iter\n",
        elapsed_per_iter(t_na.elapsed()).as_nanos()
    );

    // ─── 7. Norms Benchmarks ───
    let host_norm: Vec<f32> = (0..LINALG_LEN)
        .map(|i| (i as f32 * 0.23 - 30.0) * 1e-2)
        .collect();
    let leto_norm = leto::Array::from_shape_vec([LINALG_LEN], host_norm.clone()).unwrap();
    let nd_norm = NdArray1::from_vec(host_norm.clone());
    let na_norm = DVector::from_column_slice(&host_norm);

    let gpu_norm = device.upload(&host_norm).unwrap();
    let layout_norm = leto::Layout::c_contiguous([LINALG_LEN]).unwrap();
    let operand_norm = StridedOperand {
        buffer: &gpu_norm,
        layout: &layout_norm,
    };

    // Norm L1
    let leto_l1 = leto_ops::norm_l1(&leto_norm.view()).unwrap();
    let nd_l1 = nd_norm.mapv(|x| x.abs()).sum();
    let na_l1 = na_norm.iter().map(|x| x.abs()).sum::<f32>();
    let gpu_l1_buf = norm_l1(&device, operand_norm).unwrap();
    wait(&device);
    let mut got_l1 = [0.0f32; 1];
    device.download(&gpu_l1_buf, &mut got_l1).unwrap();
    assert!((got_l1[0] - leto_l1).abs() < 1e-3 * leto_l1.abs());
    assert!((got_l1[0] - nd_l1).abs() < 1e-3 * nd_l1.abs());
    assert!((got_l1[0] - na_l1).abs() < 1e-3 * na_l1.abs());

    println!("--- Benchmarking: Norm L1 (f32, N={LINALG_LEN}) ---");
    let t_gpu = Instant::now();
    for _ in 0..ITERS {
        let _res = norm_l1(&device, operand_norm).unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu.elapsed()).as_nanos()
    );

    let t_leto = Instant::now();
    for _ in 0..ITERS {
        let _val = leto_ops::norm_l1(black_box(&leto_norm.view())).unwrap();
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto.elapsed()).as_nanos()
    );

    let t_nd = Instant::now();
    for _ in 0..ITERS {
        let _val = black_box(black_box(&nd_norm).mapv(|x| x.abs()).sum());
    }
    println!(
        "CPU (ndarray):{} ns/iter",
        elapsed_per_iter(t_nd.elapsed()).as_nanos()
    );

    let t_na = Instant::now();
    for _ in 0..ITERS {
        let _val = black_box(black_box(&na_norm).iter().map(|x| x.abs()).sum::<f32>());
    }
    println!(
        "CPU (nalgebra):{} ns/iter\n",
        elapsed_per_iter(t_na.elapsed()).as_nanos()
    );

    // Norm L2
    let leto_l2 = leto_ops::norm_l2(&leto_norm.view()).unwrap();
    let nd_l2 = nd_norm.mapv(|x| x * x).sum().sqrt();
    let na_l2 = na_norm.norm();
    let gpu_l2_buf = norm_l2(&device, operand_norm).unwrap();
    wait(&device);
    let mut got_l2 = [0.0f32; 1];
    device.download(&gpu_l2_buf, &mut got_l2).unwrap();
    assert!((got_l2[0] - leto_l2).abs() < 1e-3 * leto_l2.abs());
    assert!((got_l2[0] - nd_l2).abs() < 1e-3 * nd_l2.abs());
    assert!((got_l2[0] - na_l2).abs() < 1e-3 * na_l2.abs());

    println!("--- Benchmarking: Norm L2 (f32, N={LINALG_LEN}) ---");
    let t_gpu = Instant::now();
    for _ in 0..ITERS {
        let _res = norm_l2(&device, operand_norm).unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu.elapsed()).as_nanos()
    );

    let t_leto = Instant::now();
    for _ in 0..ITERS {
        let _val = leto_ops::norm_l2(black_box(&leto_norm.view())).unwrap();
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto.elapsed()).as_nanos()
    );

    let t_nd = Instant::now();
    for _ in 0..ITERS {
        let _val = black_box(black_box(&nd_norm).mapv(|x| x * x).sum().sqrt());
    }
    println!(
        "CPU (ndarray):{} ns/iter",
        elapsed_per_iter(t_nd.elapsed()).as_nanos()
    );

    let t_na = Instant::now();
    for _ in 0..ITERS {
        let _val = black_box(black_box(&na_norm).norm());
    }
    println!(
        "CPU (nalgebra):{} ns/iter\n",
        elapsed_per_iter(t_na.elapsed()).as_nanos()
    );

    // Norm Max
    let leto_max = leto_ops::norm_max(&leto_norm.view()).unwrap();
    let nd_max = nd_norm.fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let na_max = na_norm
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, |acc, x| acc.max(x));
    let gpu_max_buf = norm_max(&device, operand_norm).unwrap();
    wait(&device);
    let mut got_max = [0.0f32; 1];
    device.download(&gpu_max_buf, &mut got_max).unwrap();
    assert!((got_max[0] - leto_max).abs() < 1e-3 * leto_max.abs());
    assert!((got_max[0] - nd_max).abs() < 1e-3 * nd_max.abs());
    assert!((got_max[0] - na_max).abs() < 1e-3 * na_max.abs());

    println!("--- Benchmarking: Norm Max (f32, N={LINALG_LEN}) ---");
    let t_gpu = Instant::now();
    for _ in 0..ITERS {
        let _res = norm_max(&device, operand_norm).unwrap();
    }
    wait(&device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_gpu.elapsed()).as_nanos()
    );

    let t_leto = Instant::now();
    for _ in 0..ITERS {
        let _val = leto_ops::norm_max(black_box(&leto_norm.view())).unwrap();
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto.elapsed()).as_nanos()
    );

    let t_nd = Instant::now();
    for _ in 0..ITERS {
        let _val = black_box(black_box(&nd_norm).fold(0.0f32, |acc, &x| acc.max(x.abs())));
    }
    println!(
        "CPU (ndarray):{} ns/iter",
        elapsed_per_iter(t_nd.elapsed()).as_nanos()
    );

    let t_na = Instant::now();
    for _ in 0..ITERS {
        let _val = black_box(
            black_box(&na_norm)
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, |acc, x| acc.max(x)),
        );
    }
    println!(
        "CPU (nalgebra):{} ns/iter\n",
        elapsed_per_iter(t_na.elapsed()).as_nanos()
    );
}
