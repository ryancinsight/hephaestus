//! Direct-schedule QR tests: the shared [`blocked_qr`](hephaestus_core::blocked_qr)
//! loop driven without the router.

use super::panel::GpuQrDecomposition;
use crate::infrastructure::device::WgpuDevice;
use hephaestus_core::{BlockedDecompositionBackend, ComputeDevice, blocked_qr};

fn device_or_skip() -> Option<WgpuDevice> {
    match WgpuDevice::try_default("hephaestus-qr-schedule-test") {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping WGPU QR schedule test: {error}");
            None
        }
    }
}

/// The device schedule accumulates an orthogonal **Q** independent of the
/// router: 129 columns span five 32-wide panels, so the loop walks multiple
/// panels with a trailing update each, while 160 rows keep the m×m Gram
/// check cheap. The contract suite's 2048-row case keeps elementwise
/// agreement against the host reference (which exercises the multi-tile
/// accumulate path) but no longer pays the 2048³ Gram; this clause owns
/// schedule orthogonality instead.
#[test]
fn blocked_schedule_accumulates_orthogonal_q() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let (m, n, block_size) = (160usize, 129usize, 32usize);
    assert!(n.div_ceil(block_size) > 4);

    let mut matrix_host = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            matrix_host[row * n + col] = if row == col {
                5.0
            } else {
                0.01 / (1.0 + row.abs_diff(col) as f32)
            };
        }
    }
    let matrix = device.upload(&matrix_host).expect("invariant: upload");

    // Same wrap as `qr_decompose_blocked` past the router: clone the input,
    // run the shared loop, pack the factors into the GPU decomposition.
    let work = device
        .clone_device(&matrix, m * n)
        .expect("invariant: clone");
    let result = blocked_qr(&device, work, m, n, block_size).expect("invariant: factor");
    let decomp = GpuQrDecomposition {
        inner: leto_ops::QrDecomposition::from_raw_parts(
            result.packed,
            result.heads,
            result.betas,
            m,
            n,
        ),
        r: result.r,
        rows: m,
        cols: n,
    };
    assert_eq!(decomp.shape(), (m, n));

    let q_buffer = decomp.accumulate_q(&device).expect("invariant: accumulate");
    let mut q = vec![0.0f32; m * m];
    device
        .download(&q_buffer, &mut q)
        .expect("invariant: download");

    // Elementwise agreement with a same-order reference cannot detect a wrong
    // reflector order if both orders were wrong alike, so QᵀQ is checked
    // against I independently. Accumulating a product of computed Householder
    // reflectors is backward stable with ‖Q̂ − Q‖ ≤ c(m, n)·ε, c(m, n) ≤
    // m·min(m, n) (Higham, *Accuracy and Stability*, ch. 19); the f64 Gram
    // below contributes at most O(m·ε_f64), negligible against it.
    let tolerance = 2.0 * (m * n.min(m)) as f64 * f64::from(f32::EPSILON);
    let mut max_residual = 0.0f64;
    for i in 0..m {
        for j in 0..m {
            let mut dot = 0.0f64;
            for k in 0..m {
                dot += f64::from(q[k * m + i]) * f64::from(q[k * m + j]);
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            max_residual = max_residual.max((dot - expected).abs());
        }
    }
    assert!(
        max_residual <= tolerance,
        "schedule Q is not orthogonal at [{m}, {n}]: max |QᵀQ − I| = {max_residual} exceeds {tolerance}"
    );
}
