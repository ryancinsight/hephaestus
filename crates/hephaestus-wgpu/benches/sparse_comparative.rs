//! Focused sparse CSR benchmark: Hephaestus WGPU kernels against Leto CPU.

use std::hint::black_box;
use std::time::{Duration, Instant};

use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::{spmm, spmv, GpuCsrMatrix, StridedOperand, WgpuDevice};

const ITERS: usize = 50;
const ROWS: usize = 1000;
const COLS: usize = 1000;
const RHS_COLS: usize = 128;

fn wait_wgpu(device: &WgpuDevice) {
    device
        .inner()
        .poll(wgpu::PollType::Wait)
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

    let t_wgpu = Instant::now();
    for _ in 0..ITERS {
        let out = spmv(device, &gpu_csr, &x_wg).expect("invariant: WGPU SpMV dispatch succeeds");
        black_box(out);
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

    let t_wgpu = Instant::now();
    for _ in 0..ITERS {
        let out = spmm(device, &gpu_csr, &b_op).expect("invariant: WGPU SpMM dispatch succeeds");
        black_box(out);
    }
    wait_wgpu(device);
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
    bench_spmm(&device, &cpu_csr);
}
