use super::*;
use crate::domain::decomposition::factor_cholesky_panel;
use crate::domain::error::Result;

/// Spec for the trailing SYRK update of one blocked-Cholesky step:
/// **A₂₂ -= L₂₁ · L₂₁ᵀ** inside the same device buffer.
///
/// Both operands are submatrices of one row-major buffer: the trailing block at
/// `trail_offset`, and the just-factored panel column at `panel_offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrailingSyrk {
    /// Rows (and columns) of the square trailing block.
    pub trail_extent: usize,
    /// Row stride of the containing matrix (elements per row).
    pub matrix_stride: usize,
    /// Element offset of the trailing block.
    pub trail_offset: usize,
    /// Columns in the factored panel (`b`).
    pub panel_cols: usize,
    /// Element offset of the factored panel column.
    pub panel_offset: usize,
    /// Row stride of the factored panel column.
    pub panel_stride: usize,
}

/// Backend operations the blocked-Cholesky loop needs beyond the region
/// transfers.
///
/// Blocked Cholesky adds exactly one operation kind to
/// [`BlockedDecompositionBackend`]: the trailing rank-`b` SYRK update. Unlike
/// blocked QR it needs no extra device buffer, because the SYRK reads both
/// operands from the matrix buffer itself.
pub trait BlockedCholeskyBackend: BlockedDecompositionBackend {
    /// Trailing update of one blocked-Cholesky step: **A₂₂ -= L₂₁ · L₂₁ᵀ**.
    ///
    /// Both operands are submatrices of `matrix` — the factored panel column at
    /// `panel_offset` and the trailing block at `trail_offset` — so one buffer
    /// carries the whole update.
    fn syrk_trailing(&self, matrix: &Self::Buffer, spec: TrailingSyrk) -> Result<()>;
}

/// Result of [`blocked_cholesky`]: the device-resident factor and its diagonal.
pub struct BlockedCholeskyFactors<Buffer> {
    /// Device-resident lower factor **L**; its strict upper triangle is stale
    /// until the caller's own finishing pass clears it.
    pub lower: Buffer,
    /// The factor's diagonal, retained from the host panel factorisation.
    pub diagonal: Vec<f32>,
}

/// Shared host-orchestration loop of the blocked Cholesky factorization.
///
/// Processes the `n × n` device-resident `lower` buffer in `block_size ×
/// block_size` panels. For each panel starting at row/column `k`:
///
/// 1. The compact panel column `A[k..n, k..k+b]` is gathered to the host.
/// 2. [`factor_cholesky_panel`] factors the `b × b` diagonal block and solves
///    `L₂₁` against it — the backend-neutral shared computation.
/// 3. The panel's diagonal is retained, and the factored panel is scattered
///    back.
/// 4. The trailing block is updated on the device via
///    [`BlockedCholeskyBackend::syrk_trailing`].
///
/// The strict upper triangle of the buffer is outside this loop: the panels
/// only ever write `row >= col`, so whatever the caller's finishing pass does
/// about the input's upper triangle stays its own concern. The device transfer
/// buffer is allocated once above the loop (the ADR-0003 scratch-reuse
/// discipline). `n` must be non-zero; the caller handles the empty case.
///
/// # Errors
///
/// Returns [`factor_cholesky_panel`]'s error (non-finite entry or a
/// non-positive-definite diagonal block), or the backend's transfer/launch
/// error.
pub fn blocked_cholesky<B: BlockedCholeskyBackend>(
    backend: &B,
    lower: B::Buffer,
    n: usize,
    block_size: usize,
) -> Result<BlockedCholeskyFactors<B::Buffer>> {
    debug_assert!(n > 0, "blocked_cholesky requires a non-zero dimension");
    debug_assert!(
        block_size > 0,
        "blocked_cholesky requires a non-zero block size"
    );

    let mut diagonal = vec![0.0f32; n];
    let mut panel: Vec<f32> = Vec::with_capacity(n * block_size);
    let scratch = backend.alloc(n * block_size)?;

    for k in (0..n).step_by(block_size) {
        let b = block_size.min(n - k);
        let panel_rows = n - k;
        let trail_rows = n - k - b;

        // Gather the active panel column A[k..n, k..k+b].
        let panel_region = PanelRegion {
            stride: n,
            row0: k,
            col0: k,
            rows: panel_rows,
            cols: b,
        };
        backend.download_region(&lower, panel_region, &scratch, &mut panel)?;

        factor_cholesky_panel(&mut panel, b, trail_rows)?;

        // The panel is `panel_rows × b` compact, so global cell `(k + j, k + j)`
        // sits at panel row `j`, column `j`.
        for j in 0..b {
            diagonal[k + j] = panel[j * b + j];
        }

        backend.write_region(&lower, panel_region, &scratch, &panel)?;

        if trail_rows == 0 {
            continue;
        }

        backend.syrk_trailing(
            &lower,
            TrailingSyrk {
                trail_extent: trail_rows,
                matrix_stride: n,
                trail_offset: (k + b) * n + (k + b),
                panel_cols: b,
                panel_offset: (k + b) * n + k,
                panel_stride: n,
            },
        )?;
    }

    Ok(BlockedCholeskyFactors { lower, diagonal })
}
