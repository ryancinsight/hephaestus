//! Backend-abstracted blocked decomposition loops.
//!
//! The blocked `*_decompose_blocked` entry points of the wgpu and CUDA
//! backends share their host orchestration line-for-line: the panel loop, the
//! CPU panel factorization, the permutation/sign bookkeeping, and the
//! per-panel region index math. Only three operation *kinds* are
//! backend-specific — the device→device startup copy, the compact region
//! gather/scatter between device and host, and the trailing-matrix update
//! kernel. This module hoists that shared loop into a single generic
//! [`blocked_lu`] over the [`BlockedDecompositionBackend`] trait (ADR-0003).
//!
//! The loop owns all host bookkeeping; each backend implements the trait by
//! wrapping its existing region-transfer and trailing-kernel functions. The
//! compact device transfer buffer is allocated once above the loop and passed
//! through the region calls, so a backend that stages through it (wgpu) reuses
//! one device allocation per call instead of allocating per panel; a backend
//! that transfers via pinned host staging (CUDA) ignores it.

use crate::domain::decomposition::factor_lu_panel;
use crate::domain::error::Result;

/// A rectangular region of a row-major matrix on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanelRegion {
    /// Row stride of the containing matrix (elements per row).
    pub stride: usize,
    /// First row of the region.
    pub row0: usize,
    /// First column of the region.
    pub col0: usize,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

/// Spec for the trailing GEMM update of one blocked-LU step: **C -= A · B**,
/// where A is `a_rows × a_cols`, B is `a_cols × b_cols`, C is
/// `a_rows × b_cols` — all submatrices within the same device buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrailingGemm {
    /// Element offset of the A submatrix.
    pub a_offset: usize,
    /// Row stride of A (elements per row).
    pub a_stride: usize,
    /// A row count (`m`).
    pub a_rows: usize,
    /// A column count (`k`, the GEMM inner dimension).
    pub a_cols: usize,
    /// Element offset of the B submatrix.
    pub b_offset: usize,
    /// Row stride of B.
    pub b_stride: usize,
    /// B column count (`n`).
    pub b_cols: usize,
    /// Element offset of the C submatrix.
    pub c_offset: usize,
    /// Row stride of C.
    pub c_stride: usize,
}

/// Backend operations the blocked decomposition loops need.
///
/// `Buffer` is the backend's device `f32` buffer. The loop owns all host
/// bookkeeping; implementors wrap their existing region-transfer and
/// trailing-update functions. A backend whose region transfers stage through a
/// compact device buffer may use `scratch` (allocated once by the loop);
/// backends that transfer directly (pinned host staging) may ignore it.
pub trait BlockedDecompositionBackend {
    /// The backend's device buffer for `f32` data.
    type Buffer;

    /// Allocate an `len`-element device buffer, zero-initialized.
    fn alloc(&self, len: usize) -> Result<Self::Buffer>;

    /// Copy the whole `src` into a fresh working buffer.
    ///
    /// `len` is the element count of `src`.
    fn clone_device(&self, src: &Self::Buffer, len: usize) -> Result<Self::Buffer>;

    /// Gather a compact row-major `region` of `buf` into `out` (host),
    /// resizing `out` to `region.rows * region.cols`.
    ///
    /// `out`'s existing allocation is reused, so a caller looping over panels
    /// allocates the host buffer once and refills it each iteration.
    fn download_region(
        &self,
        buf: &Self::Buffer,
        region: PanelRegion,
        scratch: &Self::Buffer,
        out: &mut Vec<f32>,
    ) -> Result<()>;

    /// Scatter a compact row-major `data` into `region` of `buf`.
    fn write_region(
        &self,
        buf: &Self::Buffer,
        region: PanelRegion,
        scratch: &Self::Buffer,
        data: &[f32],
    ) -> Result<()>;

    /// Trailing update of one blocked-LU step: **C -= A · B** inside `buf`.
    fn gemm_trailing(&self, buf: &Self::Buffer, spec: TrailingGemm) -> Result<()>;
}

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
