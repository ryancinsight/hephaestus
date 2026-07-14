//! Focused sparse CSR benchmark: Hephaestus WGPU kernels against Leto CPU.

use std::hint::black_box;
use std::time::{Duration, Instant};

use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::{
    GpuCsrMatrix, PreparedSparseDispatch, StridedOperand, WgpuDevice, prepare_spmm, prepare_spmv,
    prepare_spmv_many, spmm, spmv, spmv_many, submit_prepared_sparse_batch,
};

const ITERS: usize = 50;
const ROWS: usize = 1000;
const COLS: usize = 1000;
const RHS_COLS: usize = 128;

fn wait_wgpu(device: &WgpuDevice) {
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("invariant: benchmark device poll succeeds");
}

fn elapsed_per_iter(elapsed: Duration) -> Duration {
    elapsed / u32::try_from(ITERS).expect("invariant: iteration count fits u32")
}

fn tridiagonal_csr() -> leto_ops::CsrMatrix<f32> {
    let mut dense_host = vec![0.0f32; ROWS * COLS];
    for row in 0..ROWS {
        dense_host[row * COLS + row] = 2.0;
        if row > 0 {
            dense_host[row * COLS + row - 1] = -1.0;
        }
        if row + 1 < COLS {
            dense_host[row * COLS + row + 1] = -1.0;
        }
    }
    let layout = leto::Layout::c_contiguous([ROWS, COLS])
        .expect("invariant: benchmark dense layout is valid");
    leto_ops::CsrMatrix::from_dense(&leto::ArrayView2::new(layout, &dense_host))
}

fn assert_close_slice(got: &[f32], expected: &[f32], abs_tol: f32) {
    assert_eq!(got.len(), expected.len());
    for (index, (&got, &expected)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (got - expected).abs();
        assert!(
            diff <= abs_tol,
            "index {index}: got {got}, expected {expected}, diff {diff}, tol {abs_tol}"
        );
    }
}

fn bench_spmv(device: &WgpuDevice, cpu_csr: &leto_ops::CsrMatrix<f32>) {
    println!("--- Benchmarking: SpMV (f32, {ROWS}x{COLS} CSR) ---");

    let x_host = vec![1.0f32; COLS];
    let x_leto = leto::Array::from_shape_vec([COLS], x_host.clone())
        .expect("invariant: vector layout is valid");
    let expected = leto_ops::spmv(cpu_csr, &x_leto.view()).expect("invariant: Leto SpMV succeeds");

    let t_leto = Instant::now();
    for _ in 0..ITERS {
        let out = leto_ops::spmv(black_box(cpu_csr), black_box(&x_leto.view()))
            .expect("invariant: Leto SpMV succeeds");
        black_box(out);
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto.elapsed()).as_nanos()
    );

    let gpu_csr = GpuCsrMatrix::from_cpu(device, cpu_csr).expect("invariant: CSR upload succeeds");
    let x_wg = device
        .upload(&x_host)
        .expect("invariant: x upload succeeds");
    let y_wg = spmv(device, &gpu_csr, &x_wg).expect("invariant: WGPU SpMV dispatch succeeds");
    wait_wgpu(device);
    let mut got = vec![0.0f32; ROWS];
    device
        .download(&y_wg, &mut got)
        .expect("invariant: WGPU SpMV download succeeds");
    assert_close_slice(&got, leto::Storage::as_slice(expected.storage()), 1.0e-4);

    let mut y_reused = device
        .alloc_zeroed::<f32>(ROWS)
        .expect("invariant: reusable SpMV output allocation succeeds");
    let prepared_spmv = prepare_spmv(device, &gpu_csr, &x_wg, &mut y_reused)
        .expect("invariant: WGPU prepared SpMV succeeds");
    let t_wgpu = Instant::now();
    for _ in 0..ITERS {
        prepared_spmv.dispatch();
        black_box(&y_reused);
    }
    wait_wgpu(device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
    );
    println!();
}

fn bench_vector_batched_spmv(device: &WgpuDevice, cpu_csr: &leto_ops::CsrMatrix<f32>) {
    println!(
        "--- Benchmarking: Batched SpMV via SpMM (f32, {ROWS}x{COLS} CSR * {RHS_COLS} RHS vectors) ---"
    );

    let b_host: Vec<f32> = (0..COLS * RHS_COLS)
        .map(|index| index as f32 * 0.01 + 0.5)
        .collect();
    let b_leto = leto::Array::from_shape_vec([COLS, RHS_COLS], b_host.clone())
        .expect("invariant: dense RHS layout is valid");
    let expected = leto_ops::spmm(cpu_csr, &b_leto.view())
        .expect("invariant: Leto batched SpMV/SpMM succeeds");
    let x_columns = (0..RHS_COLS)
        .map(|col| {
            let column = (0..COLS)
                .map(|row| b_host[row * RHS_COLS + col])
                .collect::<Vec<_>>();
            leto::Array::from_shape_vec([COLS], column)
                .expect("invariant: vector RHS layout is valid")
        })
        .collect::<Vec<_>>();

    let t_leto = Instant::now();
    for _ in 0..ITERS {
        for x in &x_columns {
            let out = leto_ops::spmv(black_box(cpu_csr), black_box(&x.view()))
                .expect("invariant: Leto SpMV succeeds");
            black_box(out);
        }
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto.elapsed()).as_nanos()
    );

    let gpu_csr = GpuCsrMatrix::from_cpu(device, cpu_csr).expect("invariant: CSR upload succeeds");
    let b_wg = device
        .upload(&b_host)
        .expect("invariant: RHS batch upload succeeds");
    let b_layout =
        leto::Layout::c_contiguous([COLS, RHS_COLS]).expect("invariant: RHS layout is valid");
    let b_op = StridedOperand {
        buffer: &b_wg,
        layout: &b_layout,
    };
    let c_wg =
        spmv_many(device, &gpu_csr, &b_op).expect("invariant: WGPU batched SpMV dispatch succeeds");
    wait_wgpu(device);
    let mut got = vec![0.0f32; ROWS * RHS_COLS];
    device
        .download(&c_wg, &mut got)
        .expect("invariant: WGPU batched SpMV download succeeds");
    assert_close_slice(&got, leto::Storage::as_slice(expected.storage()), 1.0e-3);

    let mut c_reused = device
        .alloc_zeroed::<f32>(ROWS * RHS_COLS)
        .expect("invariant: reusable batched SpMV output allocation succeeds");
    let prepared_spmm = prepare_spmv_many(device, &gpu_csr, &b_op, &mut c_reused)
        .expect("invariant: WGPU prepared batched SpMV succeeds");
    prepared_spmm.dispatch();
    wait_wgpu(device);
    let mut got_reused = vec![0.0f32; ROWS * RHS_COLS];
    device
        .download(&c_reused, &mut got_reused)
        .expect("invariant: WGPU batched SpMV download succeeds");
    assert_close_slice(
        &got_reused,
        leto::Storage::as_slice(expected.storage()),
        1.0e-3,
    );

    let t_wgpu = Instant::now();
    for _ in 0..ITERS {
        prepared_spmm.dispatch();
        black_box(&c_reused);
    }
    wait_wgpu(device);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
    );
    println!();
}

fn bench_spmm(device: &WgpuDevice, cpu_csr: &leto_ops::CsrMatrix<f32>) {
    println!("--- Benchmarking: SpMM (f32, {ROWS}x{COLS} CSR * {COLS}x{RHS_COLS} dense) ---");

    let b_host: Vec<f32> = (0..COLS * RHS_COLS)
        .map(|index| index as f32 * 0.01 + 0.5)
        .collect();
    let b_leto = leto::Array::from_shape_vec([COLS, RHS_COLS], b_host.clone())
        .expect("invariant: dense rhs layout is valid");
    let expected = leto_ops::spmm(cpu_csr, &b_leto.view()).expect("invariant: Leto SpMM succeeds");

    let t_leto = Instant::now();
    for _ in 0..ITERS {
        let out = leto_ops::spmm(black_box(cpu_csr), black_box(&b_leto.view()))
            .expect("invariant: Leto SpMM succeeds");
        black_box(out);
    }
    println!(
        "CPU (Leto):   {} ns/iter",
        elapsed_per_iter(t_leto.elapsed()).as_nanos()
    );

    let gpu_csr = GpuCsrMatrix::from_cpu(device, cpu_csr).expect("invariant: CSR upload succeeds");
    let b_wg = device
        .upload(&b_host)
        .expect("invariant: rhs upload succeeds");
    let b_layout =
        leto::Layout::c_contiguous([COLS, RHS_COLS]).expect("invariant: dense rhs layout is valid");
    let b_op = StridedOperand {
        buffer: &b_wg,
        layout: &b_layout,
    };
    let c_wg = spmm(device, &gpu_csr, &b_op).expect("invariant: WGPU SpMM dispatch succeeds");
    wait_wgpu(device);
    let mut got = vec![0.0f32; ROWS * RHS_COLS];
    device
        .download(&c_wg, &mut got)
        .expect("invariant: WGPU SpMM download succeeds");
    assert_close_slice(&got, leto::Storage::as_slice(expected.storage()), 1.0e-3);

    let mut c_reused = device
        .alloc_zeroed::<f32>(ROWS * RHS_COLS)
        .expect("invariant: reusable SpMM output allocation succeeds");
    let prepared_spmm = prepare_spmm(device, &gpu_csr, &b_op, &mut c_reused)
        .expect("invariant: WGPU prepared SpMM succeeds");
    let mut c_batch_outputs = Vec::with_capacity(ITERS);
    let mut spmm_prepared = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let mut c = device
            .alloc_zeroed::<f32>(ROWS * RHS_COLS)
            .expect("invariant: batched SpMM output allocation succeeds");
        spmm_prepared.push(
            prepare_spmm(device, &gpu_csr, &b_op, &mut c)
                .expect("invariant: WGPU batched SpMM preparation succeeds"),
        );
        c_batch_outputs.push(c);
    }
    let spmm_batch = spmm_prepared
        .iter()
        .map(PreparedSparseDispatch::Spmm)
        .collect::<Vec<_>>();
    submit_prepared_sparse_batch(&spmm_batch)
        .expect("invariant: WGPU batched SpMM warm-up succeeds");
    wait_wgpu(device);
    let t_wgpu = Instant::now();
    submit_prepared_sparse_batch(&spmm_batch).expect("invariant: WGPU batched SpMM succeeds");
    wait_wgpu(device);
    black_box(&c_batch_outputs);
    black_box(&prepared_spmm);
    println!(
        "GPU (WGPU):   {} ns/iter",
        elapsed_per_iter(t_wgpu.elapsed()).as_nanos()
    );
    println!();
}

fn main() {
    println!("=== Starting sparse comparative benchmarks (WGPU vs Leto) ===");
    println!("Iterations: {ITERS}");

    let device = WgpuDevice::try_default("hephaestus-wgpu-sparse-comparative")
        .expect("WGPU GPU Backend unavailable");
    let cpu_csr = tridiagonal_csr();
    bench_spmv(&device, &cpu_csr);
    bench_vector_batched_spmv(&device, &cpu_csr);
    bench_spmm(&device, &cpu_csr);
}
