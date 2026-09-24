use super::*;
use crate::domain::decomposition::{apply_packed_qr_panel_left, panel_qr_packed};
use crate::domain::error::Result;

/// Spec for the trailing Householder application of one blocked-QR step.
///
/// The loop owns the packed reflector vectors and every host-side bookkeeping
/// value; a backend maps the panel/trailing extents and the per-reflector
/// offsets and `β` coefficients onto its trailing-application kernel.
#[derive(Clone, Copy, Debug)]
pub struct TrailingHh<'a> {
    /// Rows in the factored panel (`m − k`).
    pub panel_rows: usize,
    /// Trailing columns the reflectors are applied to (`n − k − b`).
    pub trail_cols: usize,
    /// Row stride of the containing matrix (elements per row).
    pub matrix_cols: usize,
    /// First column of the panel (`k`).
    pub panel_start: usize,
    /// Element offset of each packed reflector vector in the vectors buffer.
    pub vector_offsets: &'a [usize],
    /// The `β` coefficient of each reflector, in application order.
    pub betas: &'a [f32],
}

/// Backend operations the blocked-QR loop needs beyond the region transfers.
///
/// [`BlockedDecompositionBackend`] covers the startup copy, the region
/// gather/scatter, and the trailing GEMM of the blocked-LU loop. Blocked QR
/// adds one operation kind — the trailing Householder application — which
/// needs a second device buffer for the reflector metadata: a backend-private
/// representation the core loop cannot build. This trait declares that buffer
/// as an associated type and the calls that touch it.
///
/// It is a *separate* trait rather than a fourth method on
/// [`BlockedDecompositionBackend`] so a backend adopts the blocked-QR loop
/// without changing its blocked-LU impl: adding a method or an associated type
/// to the base trait is a breaking change for every implementor, including the
/// backends whose blocked-QR entry point has not migrated yet.
pub trait BlockedQrBackend: BlockedDecompositionBackend {
    /// The backend's reflector-metadata buffer for the trailing update.
    type Reflectors;

    /// Allocate an `len`-reflector metadata buffer.
    fn alloc_reflectors(&self, len: usize) -> Result<Self::Reflectors>;

    /// Copy `data` into `buf` from element zero.
    ///
    /// The loop packs each panel's Householder vectors into one flat host
    /// buffer before every trailing update. This is the flat refill of the
    /// vectors buffer, separate from `write_region` because a packed-vector
    /// buffer is not a matrix region.
    fn write_flat(&self, buf: &Self::Buffer, data: &[f32]) -> Result<()>;

    /// Apply the packed Householder reflectors to `matrix`'s trailing columns.
    fn householder_trailing(
        &self,
        vectors: &Self::Buffer,
        matrix: &Self::Buffer,
        reflectors: &Self::Reflectors,
        spec: TrailingHh<'_>,
    ) -> Result<()>;

    /// Whether the loop finishes a final `≤ block_size`-wide tail on the host.
    ///
    /// A backend whose trailing kernel has a fixed launch cost can apply the
    /// last panel's reflectors with [`apply_packed_qr_panel_left`] and factor
    /// the tail with [`panel_qr_packed`] on the CPU, replacing one device
    /// dispatch with host work the loop already holds the data for. Backends
    /// whose trailing update is relatively cheaper keep the tail on the device
    /// by leaving the default. The result is the same either way; only the
    /// schedule differs.
    fn finishes_tail_on_cpu(&self) -> bool {
        false
    }
}

/// Result of [`blocked_qr`]: the device-resident factor plus host bookkeeping.
pub struct BlockedQrFactors<Buffer> {
    /// Device-resident factored matrix; its upper triangle is **R**.
    pub r: Buffer,
    /// Host-side packed matrix: **R** above the diagonal, reflector tails below.
    pub packed: Vec<f32>,
    /// Cumulative Householder heads, in panel order.
    pub heads: Vec<f32>,
    /// Cumulative Householder `β` coefficients, in panel order.
    pub betas: Vec<f32>,
}

/// Shared host-orchestration loop of the blocked QR factorization.
///
/// Processes the `m × n` device-resident `work` buffer in `block_size ×
/// block_size` column panels. For each panel starting at column `k`:
///
/// 1. The compact column panel `A[k..m, k..k+b]` is gathered to the host.
/// 2. [`panel_qr_packed`] factors it and returns the panel's Householder heads
///    and `β` coefficients.
/// 3. The packed reflector vectors are extracted and the strictly-lower
///    triangle is zeroed, then the panel is scattered back.
/// 4. The `b` reflectors are applied to the trailing columns
///    `A[k..m, k+b..n]` on the device via
///    [`BlockedQrBackend::householder_trailing`].
///
/// The per-panel gathers carry only the sub-diagonal reflector tails, so **R**'s
/// upper triangle comes from one whole-matrix gather after the loop. The host
/// scratch buffers and the device transfer buffer are allocated once above the
/// loop and refilled each iteration (the ADR-0003 scratch-reuse discipline).
/// `m` must be at least `n` and `n` non-zero; the caller handles the empty case
/// before calling this loop.
///
/// # Errors
///
/// Returns [`panel_qr_packed`]'s error (non-finite entry or zero column norm),
/// or the backend's transfer/launch error.
pub fn blocked_qr<B: BlockedQrBackend>(
    backend: &B,
    work: B::Buffer,
    m: usize,
    n: usize,
    block_size: usize,
) -> Result<BlockedQrFactors<B::Buffer>> {
    debug_assert!(m >= n, "blocked_qr requires m >= n");
    debug_assert!(block_size > 0, "blocked_qr requires a non-zero block size");

    let mut packed = vec![0.0f32; m * n];
    let mut heads: Vec<f32> = Vec::with_capacity(n.min(m));
    let mut betas: Vec<f32> = Vec::with_capacity(n.min(m));

    let mut panel: Vec<f32> = Vec::with_capacity(m * block_size);
    let mut tail: Vec<f32> = Vec::with_capacity(m * block_size);
    let mut packed_vectors: Vec<f32> = Vec::with_capacity(m * block_size);
    let mut vector_offsets: Vec<usize> = Vec::with_capacity(block_size);

    let scratch = backend.alloc(m * n)?;
    let vectors = backend.alloc(m * block_size)?;
    let reflectors = backend.alloc_reflectors(block_size)?;

    for k in (0..n).step_by(block_size) {
        let b = block_size.min(n - k);
        let panel_rows = m - k;
        let trail_cols = n - k - b;

        // Gather the active column panel A[k..m, k..k+b] ((m-k) × b).
        let panel_region = PanelRegion {
            stride: n,
            row0: k,
            col0: k,
            rows: panel_rows,
            cols: b,
        };
        backend.download_region(&work, panel_region, &scratch, &mut panel)?;

        let (panel_heads, panel_betas) = panel_qr_packed(&mut panel, panel_rows, b)?;
        heads.extend_from_slice(&panel_heads);
        betas.extend_from_slice(&panel_betas);

        // A backend whose trailing kernel has a fixed launch cost finishes a
        // small final tail on the host: the panel's reflectors are applied to
        // the remaining columns, the tail is factored with the panel routine,
        // and the walk ends. This replaces one device dispatch with work the
        // loop already holds the data for and leaves the result unchanged.
        if backend.finishes_tail_on_cpu() && trail_cols > 0 && trail_cols <= block_size {
            let tail_region = PanelRegion {
                stride: n,
                row0: k,
                col0: k + b,
                rows: panel_rows,
                cols: trail_cols,
            };
            backend.download_region(&work, tail_region, &scratch, &mut tail)?;
            apply_packed_qr_panel_left(
                &panel,
                panel_rows,
                b,
                &panel_heads,
                &panel_betas,
                &mut tail,
                trail_cols,
            )?;
            let tail_rows = m - k - b;
            let (tail_heads, tail_betas) =
                panel_qr_packed(&mut tail[b * trail_cols..], tail_rows, trail_cols)?;
            heads.extend_from_slice(&tail_heads);
            betas.extend_from_slice(&tail_betas);

            // Record the strictly-lower reflector tails and clear them in one
            // pass, then scatter both region columns back.
            for j in 0..b {
                for local in (j + 1)..panel_rows {
                    let index = local * b + j;
                    packed[(k + local) * n + (k + j)] = panel[index];
                    panel[index] = 0.0;
                }
            }
            for j in 0..trail_cols {
                for local in (b + j + 1)..panel_rows {
                    let index = local * trail_cols + j;
                    packed[(k + local) * n + (k + b + j)] = tail[index];
                    tail[index] = 0.0;
                }
            }
            backend.write_region(&work, panel_region, &scratch, &panel)?;
            backend.write_region(&work, tail_region, &scratch, &tail)?;
            break;
        }

        // Record the strictly-lower reflector tails in the host packed matrix.
        for j in 0..b {
            let col = k + j;
            for row in (col + 1)..m {
                packed[row * n + col] = panel[(row - k) * b + j];
            }
        }

        // Pack the Householder vectors before the sub-diagonal is zeroed: each
        // vector is the reflector head followed by the below-diagonal column.
        packed_vectors.clear();
        vector_offsets.clear();
        for j in 0..b {
            let vector_len = panel_rows - j;
            vector_offsets.push(packed_vectors.len());
            packed_vectors.push(panel_heads[j]);
            for i in 1..vector_len {
                packed_vectors.push(panel[(j + i) * b + j]);
            }
        }

        // Zero the strictly-lower triangle so the scatter writes R's shape.
        for row in 0..panel_rows {
            for col in 0..b {
                if col < row {
                    panel[row * b + col] = 0.0;
                }
            }
        }
        backend.write_region(&work, panel_region, &scratch, &panel)?;

        if trail_cols == 0 {
            continue;
        }

        backend.write_flat(&vectors, &packed_vectors)?;
        backend.householder_trailing(
            &vectors,
            &work,
            &reflectors,
            TrailingHh {
                panel_rows,
                trail_cols,
                matrix_cols: n,
                panel_start: k,
                vector_offsets: &vector_offsets,
                betas: &panel_betas,
            },
        )?;
    }

    let full_region = PanelRegion {
        stride: n,
        row0: 0,
        col0: 0,
        rows: m,
        cols: n,
    };
    let mut host: Vec<f32> = Vec::new();
    backend.download_region(&work, full_region, &scratch, &mut host)?;
    for row in 0..m {
        for col in row..n {
            packed[row * n + col] = host[row * n + col];
        }
    }

    Ok(BlockedQrFactors {
        r: work,
        packed,
        heads,
        betas,
    })
}
