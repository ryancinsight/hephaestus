//! QR decomposition entry points.

use super::super::validate::validate_dense_operand;
use super::householder::{HhReflectorMeta, householder_trailing_update};
use super::panel::GpuQrDecomposition;
use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use hephaestus_core::{
    BlockedDecompositionBackend, BlockedQrBackend, ComputeDevice, HephaestusError, Result,
    TrailingHh, blocked_qr,
};

/// Compute the Householder QR factorization on the GPU.
///
/// The entire factorization (panel + trailing) is delegated to the host via
/// [`leto_ops`].  The result is stored on the device for downstream GPU
/// consumers.  For tall matrices where the trailing Householder application
/// should run on the GPU, prefer [`qr_decompose_blocked`].
///
/// # Errors
///
/// - Underdetermined shape (*m* < *n*).
/// - Non-finite values in the input.
/// - Exactly-zero pivot column norm (rank-deficient input).
pub fn qr_decompose(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuQrDecomposition> {
    let [rows, cols] = matrix.layout.shape();
    if rows < cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR requires m ≥ n, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;

    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);

    let qr = leto_ops::qr_decompose(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("QR decomposition failed: {e}"),
    })?;

    let r_host = qr.r();
    let r_buf = device.upload(leto::Storage::as_slice(r_host.storage()))?;

    Ok(GpuQrDecomposition {
        inner: qr,
        r: r_buf,
        rows,
        cols,
    })
}

// ---------------------------------------------------------------------------
// Entry point 2 — blocked with hybrid trailing Householder application
// ---------------------------------------------------------------------------

/// Panel block size for the blocked QR algorithm.
///
/// A value of 32 balances CPU panel factorisation cost against GPU kernel
/// launch overhead. Each panel produces *b* Householder reflectors that are
/// applied to the trailing columns in one GPU dispatch.
const QR_BLOCK_SIZE: usize = 32;

/// Bounded measured panel regime where one dense host factorization outperforms
/// blocked region transfers and per-panel synchronization.
const QR_DIRECT_PANEL_LIMIT: usize = 4;

/// Blocked QR factorization **A = Q R** with hybrid trailing Householder
/// application.
///
/// Dense matrices of at most four `QR_BLOCK_SIZE` panels use the canonical
/// [`qr_decompose`] host factorization directly. This regime has no wide
/// enough trailing work to repay the blocked path's region gather/scatter,
/// compact device scratch, and per-panel synchronization, so one dense download
/// and one `R` upload replace that schedule.
///
/// The algorithm processes the matrix in panels of `QR_BLOCK_SIZE` columns.
/// For wider matrices, each panel *k* performs:
///
/// 1. The panel `A[k:m, k:k+b]` is gathered into a contiguous device buffer
///    and downloaded to the host.
/// 2. The panel is factored on the **CPU** via inline Householder QR.
/// 3. The factored panel is uploaded to the device, and a GPU kernel
///    scatters the factored upper triangle back into the main matrix and zeroes
///    out the sub-diagonal elements.
/// 4. The *b* Householder reflectors are applied to the trailing columns
///    `A[k:m, k+b:n]` directly on the **GPU** in-place while the trailing width
///    exceeds one block.
/// 5. The final panel and at-most-one-block tail are gathered together and
///    finished on the CPU before one paired device write.
///
/// # Errors
///
/// - Underdetermined shape (*m* < *n*).
/// - Non-dense (non-C-contiguous / offset / broadcast) operand: the
///   blocked path bulk-copies the matrix storage on the device.
/// - Non-finite values in the input.
/// - Rank-deficient input (zero column norm).
pub fn qr_decompose_blocked(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuQrDecomposition> {
    let [m, n] = matrix.layout.shape();
    if m < n {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR requires m ≥ n, got shape [{m}, {n}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;
    validate_dense_operand("QR", &matrix)?;

    if m == 0 || n == 0 {
        let r_buf = device.alloc_zeroed::<f32>(0)?;
        let inner =
            leto_ops::QrDecomposition::from_raw_parts(Vec::new(), Vec::new(), Vec::new(), m, n);
        return Ok(GpuQrDecomposition {
            inner,
            r: r_buf,
            rows: m,
            cols: n,
        });
    }

    let block_size = QR_BLOCK_SIZE.min(n);
    // Component profiling shows the blocked schedule remains transfer-bound
    // through four panels. Route that bounded regime directly; wider shapes
    // keep the GPU path pending their own measured crossover.
    if n.div_ceil(block_size) <= QR_DIRECT_PANEL_LIMIT {
        return qr_decompose(device, matrix);
    }

    let work_buf = device.clone_device(matrix.buffer, m * n)?;
    let result = blocked_qr(device, work_buf, m, n, block_size)?;

    let inner =
        leto_ops::QrDecomposition::from_raw_parts(result.packed, result.heads, result.betas, m, n);

    Ok(GpuQrDecomposition {
        inner,
        r: result.r,
        rows: m,
        cols: n,
    })
}

/// The WGPU blocked-QR operations the shared [`blocked_qr`] loop drives
/// (ADR 0003).
///
/// Only the reflector-metadata buffer and the trailing Householder kernel are
/// WGPU-specific: the loop owns the panel walk, the CPU panel factorisation,
/// the reflector packing, the sub-diagonal zeroing, and the final gather. WGPU
/// also selects the host tail-finish policy, because its trailing kernel has a
/// fixed launch cost that the host path avoids for a final `≤ block_size` tail.
impl BlockedQrBackend for WgpuDevice {
    type Reflectors = WgpuBuffer<HhReflectorMeta>;

    fn alloc_reflectors(&self, len: usize) -> Result<Self::Reflectors> {
        self.alloc_uninitialized::<HhReflectorMeta>(len)
    }

    fn write_flat(&self, buf: &Self::Buffer, data: &[f32]) -> Result<()> {
        self.write_sub_buffer(buf, 0, data)
    }

    fn householder_trailing(
        &self,
        vectors: &Self::Buffer,
        matrix: &Self::Buffer,
        reflectors: &Self::Reflectors,
        spec: TrailingHh<'_>,
    ) -> Result<()> {
        householder_trailing_update(self, vectors, matrix, reflectors, spec)
    }

    fn finishes_tail_on_cpu(&self) -> bool {
        true
    }
}
