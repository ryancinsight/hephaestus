//! Synchronization-floor profiling for hybrid blocked decomposition paths.
//!
//! The blocked LU/QR algorithms already have end-to-end comparative benchmark
//! rows. This harness isolates the host/device transfer pattern those paths
//! impose at the measured benchmark shapes, so follow-up kernel work targets
//! the synchronization component rather than guessing from total time.

use std::hint::black_box;
use std::time::{Duration, Instant};

use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::WgpuDevice;

const ITERS: usize = 100;

fn wait_wgpu(device: &WgpuDevice) {
    device
        .inner()
        .poll(wgpu::PollType::Wait)
        .expect("invariant: benchmark device poll succeeds");
}

fn elapsed_per_iter(elapsed: Duration) -> Duration {
    elapsed / u32::try_from(ITERS).expect("invariant: iteration count fits u32")
}

fn assert_close_slice(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (index, (&got, &expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - expected).abs() <= f32::EPSILON,
            "sync profile mismatch at {index}: got {got}, expected {expected}"
        );
    }
}

fn profile_blocked_lu_sync(device: &WgpuDevice) {
    let n = 66usize;
    let len = n * n;
    let mut host = vec![0.0f32; len];
    for row in 0..n {
        for col in 0..n {
            host[row * n + col] = if row == col {
                n as f32 + 4.0
            } else {
                0.1 / (1.0 + row.abs_diff(col) as f32)
            };
        }
    }

    let buffer = device.upload(&host).expect("upload LU profile matrix");
    let mut out = vec![0.0f32; len];
    let l21 = vec![0.25f32; 2 * 64];
    let u12 = vec![0.125f32; 64 * 2];
    let trail = vec![0.5f32; 4];
    let trail_buf = device.upload(&trail).expect("upload LU trailing tile");
    let mut trail_out = vec![0.0f32; 4];

    let start = Instant::now();
    for _ in 0..ITERS {
        device
            .download(&buffer, &mut out)
            .expect("download LU input");
        assert_close_slice(&out, &host);

        device
            .write_buffer(&buffer, &host)
            .expect("write LU panel state");
        let l21_buf = device.upload(black_box(&l21)).expect("upload LU L21");
        let u12_buf = device.upload(black_box(&u12)).expect("upload LU U12");
        black_box((&l21_buf, &u12_buf));

        device
            .download(&trail_buf, &mut trail_out)
            .expect("download LU trailing tile");
        assert_close_slice(&trail_out, &trail);

        device
            .write_buffer(&trail_buf, &trail)
            .expect("write LU final panel tile");
    }
    wait_wgpu(device);

    println!(
        "Blocked LU 66x66 sync floor: {} ns/iter",
        elapsed_per_iter(start.elapsed()).as_nanos()
    );
}

fn profile_blocked_qr_sync(device: &WgpuDevice) {
    let rows = 70usize;
    let cols = 35usize;
    let len = rows * cols;
    let mut host = vec![0.0f32; len];
    for row in 0..rows {
        for col in 0..cols {
            host[row * cols + col] = if row == col {
                5.0
            } else {
                0.01 / (1.0 + row.abs_diff(col) as f32)
            };
        }
    }

    let buffer = device.upload(&host).expect("upload QR profile matrix");
    let mut out = vec![0.0f32; len];
    let trail_cols = 3usize;
    let trailing = vec![0.25f32; rows * trail_cols];
    let trailing_buf = device
        .upload(&trailing)
        .expect("upload QR trailing columns");
    let mut trailing_out = vec![0.0f32; trailing.len()];
    let vectors: Vec<Vec<f32>> = (0..32)
        .map(|j| vec![1.0f32 / (1.0 + j as f32); rows - j])
        .collect();
    let packed_vectors: Vec<f32> = vectors.iter().flatten().copied().collect();

    let start = Instant::now();
    for _ in 0..ITERS {
        device
            .download(&buffer, &mut out)
            .expect("download QR input");
        assert_close_slice(&out, &host);

        device
            .write_buffer(&trailing_buf, &trailing)
            .expect("write QR trailing columns");
        let uploaded_vectors = device
            .upload(black_box(&packed_vectors))
            .expect("upload packed QR reflectors");
        black_box(&uploaded_vectors);

        device
            .download(&trailing_buf, &mut trailing_out)
            .expect("download QR trailing columns");
        assert_close_slice(&trailing_out, &trailing);
    }
    wait_wgpu(device);

    println!(
        "Blocked QR 70x35 sync floor: {} ns/iter",
        elapsed_per_iter(start.elapsed()).as_nanos()
    );
}

fn main() {
    let device = match WgpuDevice::try_default("hephaestus-decomposition-sync-bench") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("Skipping decomposition sync profile: WGPU device unavailable: {error}");
            return;
        }
    };

    println!("=== Hybrid decomposition synchronization profile ===");
    println!("Iterations: {ITERS}");
    println!("WGPU GPU Backend: {}", device.backend_name());
    profile_blocked_lu_sync(&device);
    profile_blocked_qr_sync(&device);
}
