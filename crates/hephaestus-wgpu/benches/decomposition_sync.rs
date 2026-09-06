//! Component profiling for hybrid blocked decomposition paths.
//!
//! The blocked LU/QR algorithms already have end-to-end comparative benchmark
//! rows. This harness retains LU's transfer floor and validates the current QR
//! production factorization, so follow-up work targets measured components
//! rather than stale synthetic transfer models.

#![cfg(feature = "decomposition")]

use std::hint::black_box;
use std::time::{Duration, Instant};

use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::{StridedOperand, WgpuDevice, qr_decompose, qr_decompose_blocked};
use leto::Layout;

const ITERS: usize = 100;
const QR_SHAPES: [[usize; 2]; 4] = [[96, 64], [96, 65], [192, 128], [192, 129]];

fn wait_wgpu(device: &WgpuDevice) {
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
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

fn qr_source(rows: usize, cols: usize) -> Vec<f32> {
    let mut host = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            host[row * cols + col] = if row == col {
                5.0
            } else {
                0.01 / (1.0 + row.abs_diff(col) as f32)
            };
        }
    }
    host
}

fn profile_blocked_qr_end_to_end(device: &WgpuDevice, rows: usize, cols: usize) {
    let host = qr_source(rows, cols);
    let buffer = device.upload(&host).expect("upload QR profile matrix");
    let layout = Layout::c_contiguous([rows, cols]).expect("invariant: profile shape is valid");
    let checked = qr_decompose_blocked(
        device,
        StridedOperand {
            buffer: &buffer,
            layout: &layout,
        },
    )
    .expect("factor QR profile validation matrix");
    let mut actual = vec![0.0f32; rows * cols];
    device
        .download(checked.r_buffer(), &mut actual)
        .expect("download QR profile validation factor");
    let leto_matrix = leto::Array::from_shape_vec([rows, cols], host.clone())
        .expect("invariant: profile shape matches storage");
    let expected = leto_ops::qr_decompose(&leto_matrix.view())
        .expect("factor Leto QR profile matrix")
        .r();
    for (index, (&actual, &expected)) in actual
        .iter()
        .zip(leto::Storage::as_slice(expected.storage()))
        .enumerate()
    {
        let tolerance = 16.0 * f32::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "QR profile factor mismatch at {index}: got {actual}, expected {expected}"
        );
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        let decomposition = qr_decompose_blocked(
            device,
            black_box(StridedOperand {
                buffer: &buffer,
                layout: &layout,
            }),
        )
        .expect("factor QR profile matrix");
        wait_wgpu(device);
        black_box(decomposition);
    }

    println!(
        "Blocked QR {rows}x{cols} current end-to-end: {} ns/iter",
        elapsed_per_iter(start.elapsed()).as_nanos()
    );

    let checked = qr_decompose(
        device,
        StridedOperand {
            buffer: &buffer,
            layout: &layout,
        },
    )
    .expect("factor direct QR profile validation matrix");
    device
        .download(checked.r_buffer(), &mut actual)
        .expect("download direct QR profile validation factor");
    for (index, (&actual, &expected)) in actual
        .iter()
        .zip(leto::Storage::as_slice(expected.storage()))
        .enumerate()
    {
        let tolerance = 16.0 * f32::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "direct QR profile factor mismatch at {index}: got {actual}, expected {expected}"
        );
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        let decomposition = qr_decompose(
            device,
            black_box(StridedOperand {
                buffer: &buffer,
                layout: &layout,
            }),
        )
        .expect("factor direct QR profile matrix");
        wait_wgpu(device);
        black_box(decomposition);
    }

    println!(
        "Direct QR {rows}x{cols} current end-to-end: {} ns/iter",
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
    for [rows, cols] in QR_SHAPES {
        profile_blocked_qr_end_to_end(&device, rows, cols);
    }
}
