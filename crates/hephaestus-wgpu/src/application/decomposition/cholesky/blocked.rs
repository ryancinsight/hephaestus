//! Blocked Cholesky entry point: CPU panels with GPU SYRK trailing updates.

use std::sync::OnceLock;

use hephaestus_core::{
    BlockedCholeskyBackend, BlockedDecompositionBackend, ComputeDevice, Result, TrailingSyrk,
    blocked_cholesky,
};
use leto::Layout;

use super::super::validate::{validate_dense_operand, validate_square};
use crate::application::strided::StridedOperand;
use crate::infrastructure::device::WgpuDevice;

use super::factor::{BLOCK_SIZE, GpuCholesky, zero_strict_upper};
use super::syrk::syrk_trailing_update;

/// Blocked Cholesky factorization **A** = **L** **L**ᵀ with GPU-accelerated
/// trailing-matrix SYRK updates.
///
/// The algorithm processes the matrix in `BLOCK_SIZE × BLOCK_SIZE` panels.
/// For each panel *k*:
///
/// 1. The diagonal block is factored on the **CPU** via [`leto_ops`]
///    (O(b³/3)).
/// 2. The off-diagonal panel is solved on the **CPU** via triangular solve
///    (O(b²·(n−k)/2)).
/// 3. The trailing submatrix is updated on the **GPU** via a dedicated SYRK
///    kernel: `A₂₂ -= L₂₁ · L₂₁ᵀ` (O(b·(n−k)²/2)).
///
/// The SYRK trailing update is the dominant cost for large *n* and is the
/// reason this entry point exists: unlike [`super::factor::cholesky_decompose`]
/// which delegates *all* O(n³) work to the CPU, this function offloads the
/// rank-k update to the GPU compute pipeline.
///
/// # Block-size tuning
///
/// `BLOCK_SIZE` is currently a compile-time constant.  A future refinement
/// could auto-tune based on the device's preferred workgroup size and
/// available shared memory.
///
/// # Errors
///
/// - Non-square matrix.
/// - Non-dense (non-C-contiguous / offset / broadcast) operand: the
///   blocked path bulk-copies the matrix storage on the device.
/// - Non-finite values in the input.
/// - Matrix is not positive-definite.
pub fn cholesky_decompose_blocked(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuCholesky> {
    let n = validate_square(&matrix)?;
    validate_dense_operand("cholesky", &matrix)?;
    if n == 0 {
        return Ok(GpuCholesky {
            inner: OnceLock::new(),
            diagonal: Vec::new(),
            lower: device.alloc_zeroed::<f32>(0)?,
            n: 0,
        });
    }

    // Copy the (dense, validated) operand into a fresh device working buffer;
    // the shared blocked-Cholesky loop owns all host bookkeeping.
    let lower_buf = device.clone_device(matrix.buffer, n * n)?;

    let block_size = BLOCK_SIZE.min(n);
    let result = blocked_cholesky(device, lower_buf, n, block_size)?;

    // Finish the factor on the device. The per-panel scatters already wrote
    // every cell with `row >= blockstart(col)`, and the panel factorisation
    // zeroes each diagonal block's strict upper, so the only cells still
    // holding input values are strictly above the diagonal outside those
    // blocks — exactly what this pass clears. The superseded
    // `write_buffer(&lower_buf, &host)` uploaded the whole `n^2` matrix to
    // achieve the same thing, re-sending the lower triangle the device had
    // already computed.
    zero_strict_upper(device, &result.lower, n)?;

    // `inner` stays empty: with the factor complete on the device and its
    // diagonal retained, only a host-side substitution needs the `n * n`
    // array, and `host_factor` downloads it then.
    Ok(GpuCholesky {
        inner: OnceLock::new(),
        diagonal: result.diagonal,
        lower: result.lower,
        n,
    })
}

/// The WGPU blocked-Cholesky operation the shared `blocked_cholesky` loop
/// drives (ADR 0003).
///
/// Only the trailing SYRK kernel is WGPU-specific: the loop owns the panel
/// iteration, the CPU panel factorisation, the diagonal retention, and the
/// panel scatter. The SYRK reads both operands from the matrix buffer, and
/// this impl submits its own encoder because the trait hands it no batch.
impl BlockedCholeskyBackend for WgpuDevice {
    fn syrk_trailing(&self, matrix: &Self::Buffer, spec: TrailingSyrk) -> Result<()> {
        let trail_layout = Layout::try_new(
            [spec.trail_extent, spec.trail_extent],
            [spec.matrix_stride as isize, 1],
            spec.trail_offset,
        )
        .expect("invariant: submatrix layout derives from a validated parent");

        let mut encoder = self
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-cholesky-syrk-update"),
            });
        syrk_trailing_update(
            self,
            &mut encoder,
            matrix,
            &trail_layout,
            spec.panel_cols,
            spec.panel_offset,
            spec.panel_stride,
        )?;
        self.queue().submit(Some(encoder.finish()));
        Ok(())
    }
}
