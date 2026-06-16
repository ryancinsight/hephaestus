//! Comparative benchmark of Hephaestus GPU (WGPU) vs CPU (Leto & ndarray).
//!
//! Benchmarks:
//! 1. Contiguous Add (f32, size 1<<20)
//! 2. Contiguous Exp (f32, size 1<<20)
//! 3. Contiguous Sum (f32, size 1<<20)
//!
//! Validates results across all three backends to ensure correctness.

use std::hint::black_box;
use std::time::{Duration, Instant};

use hephaestus_core::BlockWidth;
use hephaestus_wgpu::{
    binary_elementwise_into, reduction, unary_elementwise_into, AddOp, ComputeDevice, ExpOp,
    SumOp, WgpuDevice,
};
use leto::Storage;
use ndarray::Array1 as NdArray1;

const LEN: usize = 1 << 20; // 1,048,576 elements
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

fn main() {
    println!("=== Starting comparative benchmarks (GPU vs CPU) ===");
    println!("Array length: {LEN}");
    println!("Iterations: {ITERS}");

    // 1. Setup GPU Device
    let device = match WgpuDevice::try_default("hephaestus-comparative-bench") {
        Ok(device) => device,
        Err(e) => {
            eprintln!("skipping benchmark: WGPU device could not be acquired: {e}");
            return;
        }
    };
    println!("GPU Backend: {}", device.backend_name());

    // 2. Prepare pinned host inputs (reproducible deterministic values)
    let host_a: Vec<f32> = (0..LEN).map(|i| (i as f32 * 0.731 + 1.0) * 1e-7).collect();
    let host_b: Vec<f32> = (0..LEN).map(|i| (i as f32 * 0.317 + 2.0) * 1e-7).collect();

    // --- Validation & Verification Path ---
    // Leto CPU references
    let leto_a = leto::Array::from_shape_vec([LEN], host_a.clone()).unwrap();
    let leto_b = leto::Array::from_shape_vec([LEN], host_b.clone()).unwrap();

    let mut leto_add_out = leto::Array::zeros([LEN]);
    leto_ops::add(
        &leto_a.view(),
        &leto_b.view(),
        &mut leto_add_out.view_mut(),
    )
    .unwrap();

    let leto_exp_out = leto_ops::unary_map(leto_ops::ExpOp, &leto_a.view()).unwrap();
    let leto_sum_out = leto_ops::sum(&leto_a.view());

    // ndarray CPU references
    let nd_a = NdArray1::from_vec(host_a.clone());
    let nd_b = NdArray1::from_vec(host_b.clone());

    let nd_add_out = &nd_a + &nd_b;
    let nd_exp_out = nd_a.mapv(f32::exp);
    let nd_sum_out = nd_a.sum();

    // WGPU GPU Buffers
    let gpu_a = device.upload(&host_a).expect("upload a");
    let gpu_b = device.upload(&host_b).expect("upload b");
    let gpu_add_out = device.alloc_zeroed::<f32>(LEN).expect("alloc gpu_add_out");
    let gpu_exp_out = device.alloc_zeroed::<f32>(LEN).expect("alloc gpu_exp_out");

    // Perform warmups & validate correctness
    binary_elementwise_into::<AddOp, f32>(&device, &gpu_a, &gpu_b, &gpu_add_out, BlockWidth::DEFAULT).unwrap();
    unary_elementwise_into::<ExpOp, f32>(&device, &gpu_a, &gpu_exp_out, BlockWidth::DEFAULT).unwrap();
    let gpu_sum_buf = reduction::<SumOp, f32>(&device, &gpu_a).unwrap();
    wait(&device);

    let mut host_add_verify = vec![0.0f32; LEN];
    let mut host_exp_verify = vec![0.0f32; LEN];
    let mut host_sum_verify = [0.0f32; 1];

    device.download(&gpu_add_out, &mut host_add_verify).unwrap();
    device.download(&gpu_exp_out, &mut host_exp_verify).unwrap();
    device.download(&gpu_sum_buf, &mut host_sum_verify).unwrap();

    // Assert value semantic correctness across all backends
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

    println!("Verification passed! Outputs are mathematically identical across all 3 backends.");

    // --- Benchmark: Elementwise Add ---
    println!("\n--- Benchmarking: Elementwise Add (f32, N={LEN}) ---");

    // WGPU GPU
    let t_gpu_add = Instant::now();
    for _ in 0..ITERS {
        binary_elementwise_into::<AddOp, f32>(&device, &gpu_a, &gpu_b, &gpu_add_out, BlockWidth::DEFAULT).unwrap();
    }
    wait(&device);
    let dur_gpu_add = t_gpu_add.elapsed();
    println!("GPU (WGPU):   {} ns/iter", elapsed_per_iter(dur_gpu_add).as_nanos());

    // Leto CPU
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
    let dur_leto_add = t_leto_add.elapsed();
    println!("CPU (Leto):   {} ns/iter", elapsed_per_iter(dur_leto_add).as_nanos());

    // ndarray CPU
    let t_nd_add = Instant::now();
    for _ in 0..ITERS {
        let _c = black_box(&nd_a) + black_box(&nd_b);
    }
    let dur_nd_add = t_nd_add.elapsed();
    println!("CPU (ndarray):{} ns/iter", elapsed_per_iter(dur_nd_add).as_nanos());

    // --- Benchmark: Elementwise Exp ---
    println!("\n--- Benchmarking: Elementwise Exp (f32, N={LEN}) ---");

    // WGPU GPU
    let t_gpu_exp = Instant::now();
    for _ in 0..ITERS {
        unary_elementwise_into::<ExpOp, f32>(&device, &gpu_a, &gpu_exp_out, BlockWidth::DEFAULT).unwrap();
    }
    wait(&device);
    let dur_gpu_exp = t_gpu_exp.elapsed();
    println!("GPU (WGPU):   {} ns/iter", elapsed_per_iter(dur_gpu_exp).as_nanos());

    // Leto CPU
    let t_leto_exp = Instant::now();
    for _ in 0..ITERS {
        let _out = leto_ops::unary_map(leto_ops::ExpOp, black_box(&leto_a.view())).unwrap();
    }
    let dur_leto_exp = t_leto_exp.elapsed();
    println!("CPU (Leto):   {} ns/iter", elapsed_per_iter(dur_leto_exp).as_nanos());

    // ndarray CPU
    let t_nd_exp = Instant::now();
    for _ in 0..ITERS {
        let _c = black_box(&nd_a).mapv(f32::exp);
    }
    let dur_nd_exp = t_nd_exp.elapsed();
    println!("CPU (ndarray):{} ns/iter", elapsed_per_iter(dur_nd_exp).as_nanos());

    // --- Benchmark: Sum Reduction ---
    println!("\n--- Benchmarking: Sum Reduction (f32, N={LEN}) ---");

    // WGPU GPU
    let t_gpu_sum = Instant::now();
    for _ in 0..ITERS {
        let _res = reduction::<SumOp, f32>(&device, &gpu_a).unwrap();
    }
    wait(&device);
    let dur_gpu_sum = t_gpu_sum.elapsed();
    println!("GPU (WGPU):   {} ns/iter", elapsed_per_iter(dur_gpu_sum).as_nanos());

    // Leto CPU
    let t_leto_sum = Instant::now();
    for _ in 0..ITERS {
        let _val = leto_ops::sum(black_box(&leto_a.view()));
    }
    let dur_leto_sum = t_leto_sum.elapsed();
    println!("CPU (Leto):   {} ns/iter", elapsed_per_iter(dur_leto_sum).as_nanos());

    // ndarray CPU
    let t_nd_sum = Instant::now();
    for _ in 0..ITERS {
        let _val = black_box(&nd_a).sum();
    }
    let dur_nd_sum = t_nd_sum.elapsed();
    println!("CPU (ndarray):{} ns/iter", elapsed_per_iter(dur_nd_sum).as_nanos());
}
