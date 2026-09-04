//! Criterion benchmarks for measured blocked-QR routing boundaries.
//!
//! Local evidence records the machine and driver in `benchmark_results.md`.
//! The 70×35 workload crosses the fixed 32-column panel boundary with a
//! three-column tail; 192×128 exercises the direct-route limit; 192×129 and
//! 384×256 exercise retained blocked schedules. Every timed iteration factors
//! the same device-resident input and waits for completion; a separate value
//! check compares the complete `R` factor against Leto before measurement.

#![cfg(feature = "decomposition")]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use hephaestus_wgpu::{ComputeDevice, StridedOperand, WgpuDevice, qr_decompose_blocked};
use leto::Layout;

fn matrix_values<const ROWS: usize, const COLS: usize>() -> Vec<f32> {
    let mut matrix = vec![0.0; ROWS * COLS];
    for row in 0..ROWS {
        for col in 0..COLS {
            matrix[row * COLS + col] = if row == col {
                5.0
            } else {
                0.01 / (1.0 + row.abs_diff(col) as f32)
            };
        }
    }
    matrix
}

fn wait(device: &WgpuDevice) {
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("invariant: benchmark device poll succeeds");
}

fn assert_r_matches_leto<const ROWS: usize, const COLS: usize>(
    device: &WgpuDevice,
    matrix: &hephaestus_wgpu::WgpuBuffer<f32>,
    layout: &Layout<2>,
    host: &[f32],
) {
    let gpu = qr_decompose_blocked(
        device,
        StridedOperand {
            buffer: matrix,
            layout,
        },
    )
    .expect("benchmark blocked QR");
    let leto_matrix = leto::Array::from_shape_vec([ROWS, COLS], host.to_vec())
        .expect("invariant: benchmark shape matches storage");
    let expected = leto_ops::qr_decompose(&leto_matrix.view())
        .expect("benchmark Leto QR")
        .r();
    let expected = leto::Storage::as_slice(expected.storage());
    let mut actual = vec![0.0; ROWS * COLS];
    device
        .download(gpu.r_buffer(), &mut actual)
        .expect("benchmark R download");

    for row in 0..ROWS {
        for col in 0..COLS {
            let expected_value = expected[row * COLS + col];
            let tolerance = 16.0 * f32::EPSILON * expected_value.abs().max(1.0);
            assert!(
                (actual[row * COLS + col] - expected_value).abs() <= tolerance,
                "R[{row},{col}] mismatch: got {}, expected {expected_value}",
                actual[row * COLS + col]
            );
        }
    }
}

fn measure_shape<const ROWS: usize, const COLS: usize>(
    c: &mut Criterion,
    device: &WgpuDevice,
    id: &str,
) {
    let host = matrix_values::<ROWS, COLS>();
    let matrix = device.upload(&host).expect("benchmark input upload");
    let layout = Layout::c_contiguous([ROWS, COLS]).expect("invariant: benchmark shape is valid");

    assert_r_matches_leto::<ROWS, COLS>(device, &matrix, &layout, &host);

    c.bench_function(id, |b| {
        b.iter(|| {
            let result = qr_decompose_blocked(
                device,
                black_box(StridedOperand {
                    buffer: &matrix,
                    layout: &layout,
                }),
            )
            .expect("benchmark blocked QR");
            wait(device);
            black_box(result)
        });
    });
}

fn blocked_qr_tail(c: &mut Criterion) {
    let device = match WgpuDevice::try_default("hephaestus-blocked-qr-tail-bench") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping WGPU benchmark: {error}");
            return;
        }
    };
    measure_shape::<70, 35>(c, &device, "blocked_qr/final_two_panels_70x35");
    measure_shape::<192, 128>(c, &device, "blocked_qr/four_panels_192x128");
    measure_shape::<192, 129>(c, &device, "blocked_qr/routing_boundary_192x129");
    measure_shape::<384, 256>(c, &device, "blocked_qr/eight_panels_384x256");
}

criterion_group!(benches, blocked_qr_tail);
criterion_main!(benches);
