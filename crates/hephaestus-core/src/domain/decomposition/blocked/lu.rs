use super::*;
use crate::domain::decomposition::factor_lu_panel;
use crate::domain::error::Result;

/// Result of [`blocked_lu`]: the device-resident factors plus the host-side
/// bookkeeping needed to build a decomposition.
pub struct BlockedLuFactors<Buffer> {
    /// Device-resident packed L/U factors (*n* × *n*, row-major).
    pub factors: Buffer,
    /// Cumulative row permutation applied to the full matrix.
    pub perm: Vec<usize>,
    /// Sign of the permutation.
    pub sign: i8,
    /// Host-side packed factor matrix (*n* × *n*, row-major).
    pub host: Vec<f32>,
}

/// Shared host-orchestration loop of the blocked LU factorization.
///
/// Processes the `n × n` device-resident `factors` buffer in
/// `block_size × block_size` panels. For each panel starting at row `k`:
///
/// 1. The column panel `A[k..n, k..k+b]` and row panel `A[k..k+b, 0..n]` are
///    gathered to the host.
/// 2. [`factor_lu_panel`] factors the diagonal block on the host and solves
///    the `L₂₁`/`U₁₂` panels (identical partial-pivoting rule to Leto's LU).
/// 3. The factored panels are scattered back to the device and the trailing
///    submatrix is updated on the device via
///    [`BlockedDecompositionBackend::gemm_trailing`] (`A₂₂ -= L₂₁ · U₁₂`).
///
/// The host scratch buffers and the compact device transfer buffer are
/// allocated once above the loop and refilled each iteration (wgpu reuse
/// discipline, ADR-0003 §Scratch-reuse). `n` must be non-zero; the caller
/// handles the empty case before calling this loop.
///
/// # Errors
///
/// Returns [`factor_lu_panel`]'s error (non-finite entry or zero pivot), or
/// the backend's transfer/launch error.
pub fn blocked_lu<B: BlockedDecompositionBackend>(
    backend: &B,
    factors: B::Buffer,
    n: usize,
    block_size: usize,
) -> Result<BlockedLuFactors<B::Buffer>> {
    debug_assert!(n > 0, "blocked_lu requires a non-zero dimension");
    debug_assert!(block_size > 0, "blocked_lu requires a non-zero block size");

    let mut perm: Vec<usize> = (0..n).collect();
    let mut sign = 1i8;
    let mut host = vec![0.0f32; n * n];

    let mut col_panel: Vec<f32> = Vec::with_capacity(n * block_size);
    let mut row_panel: Vec<f32> = Vec::with_capacity(block_size * n);
    let mut diag = vec![0.0f32; block_size * block_size];
    let scratch = backend.alloc(n * block_size)?;

    for k in (0..n).step_by(block_size) {
        let b = block_size.min(n - k);
        let trail = n - k - b;

        // Gather the active column panel A[k..n, k..k+b] ((n-k) × b).
        let col_region = PanelRegion {
            stride: n,
            row0: k,
            col0: k,
            rows: n - k,
            cols: b,
        };
        backend.download_region(&factors, col_region, &scratch, &mut col_panel)?;

        // Gather the active row panel A[k..k+b, 0..n] (b × n).
        let row_region = PanelRegion {
            stride: n,
            row0: k,
            col0: 0,
            rows: b,
            cols: n,
        };
        backend.download_region(&factors, row_region, &scratch, &mut row_panel)?;

        factor_lu_panel(
            &mut col_panel,
            &mut row_panel,
            &mut diag,
            k,
            b,
            n,
            trail,
            &mut perm,
            &mut sign,
        )?;

        // Record the finalized rows in the host-side packed factor matrix.
        for i in 0..b {
            let row = k + i;
            for j in 0..n {
                host[row * n + j] = row_panel[i * n + j];
            }
        }

        if trail == 0 {
            // Final panel: write the factored rows back and finish.
            backend.write_region(&factors, row_region, &scratch, &row_panel)?;
            continue;
        }

        let col_write_region = PanelRegion {
            stride: n,
            row0: k + b,
            col0: k,
            rows: trail,
            cols: b,
        };
        backend.write_region(&factors, col_write_region, &scratch, &col_panel[(b * b)..])?;
        backend.write_region(&factors, row_region, &scratch, &row_panel)?;

        backend.gemm_trailing(
            &factors,
            TrailingGemm {
                a_offset: (k + b) * n + k,
                a_stride: n,
                a_rows: trail,
                a_cols: b,
                b_offset: k * n + (k + b),
                b_stride: n,
                b_cols: trail,
                c_offset: (k + b) * n + (k + b),
                c_stride: n,
            },
        )?;
    }

    Ok(BlockedLuFactors {
        factors,
        perm,
        sign,
        host,
    })
}
