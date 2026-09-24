//! Shared region/backend vocabulary for the blocked decomposition loops.
//!
//! [`PanelRegion`] and [`TrailingGemm`] describe the host-orchestrated data
//! movement and trailing update every blocked loop drives, and
//! [`BlockedDecompositionBackend`] is the seam a backend implements once to
//! adopt [`blocked_lu`](super::blocked_lu); [`BlockedQrBackend`](super::qr::BlockedQrBackend)
//! and [`BlockedCholeskyBackend`](super::cholesky::BlockedCholeskyBackend)
//! extend it for their own trailing kernels.

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
