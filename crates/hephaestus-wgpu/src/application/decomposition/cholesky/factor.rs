use std::sync::OnceLock;

use hephaestus_core::{
    BlockedDecompositionBackend, ComputeDevice, HephaestusError, Result, blocked_cholesky,
};

use super::super::validate::{validate_dense_operand, validate_square};
use super::solve::device_solve;
use super::triangle::zero_strict_upper;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

// ---------------------------------------------------------------------------
// Entry point 2 — blocked with GPU trailing SYRK
// ---------------------------------------------------------------------------

/// Panel block size for the blocked Cholesky algorithm.
///
/// A value of 64 balances CPU panel factorisation cost against GPU SYRK
/// launch overhead.  For matrices smaller than `BLOCK_SIZE` the algorithm
/// degrades gracefully to a single CPU panel pass.
const BLOCK_SIZE: usize = 64;
// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Lower-triangular Cholesky factor on the device, solving on the device and
/// inverting through a host-side decomposition without re-factorization.
///
/// # Host residency
///
/// [`solve`](Self::solve) substitutes on the device against the resident
/// factor and [`det`](Self::det) reads only the factor's diagonal; the
/// `n × n` host factor serves [`inv`](Self::inv) alone.
/// [`cholesky_decompose_blocked`] therefore keeps just that diagonal (`n`
/// elements, extracted from panels it already downloads) and leaves `inner`
/// empty, materializing it from the device factor on the first `inv` and
/// caching it thereafter. [`cholesky_decompose`] factors on the host to
/// begin with, so it populates `inner` eagerly and never downloads.
pub struct GpuCholesky {
    /// Host-side leto-ops decomposition, materialized on first host-side
    /// inversion. Pre-populated by the host-delegating entry point.
    inner: OnceLock<leto_ops::CholeskyDecomposition<f32>>,
    /// Diagonal of **L** in factor order, `n` elements.
    ///
    /// Bitwise equal to the device factor's diagonal: both are written from
    /// the same host panel values.
    diagonal: Vec<f32>,
    /// Device-resident lower-triangular factor **L** (*n* × *n*, row-major).
    lower: WgpuBuffer<f32>,
    n: usize,
}

impl GpuCholesky {
    /// The host factor for [`inv`](Self::inv), downloading and caching it on first use.
    ///
    /// The device factor is authoritative and complete once
    /// [`zero_strict_upper`] has run: the per-panel scatters write every cell
    /// with `row >= blockstart(col)` from the same host panel values the old
    /// eager array held, and that pass clears the rest. Downloading it
    /// therefore reconstructs exactly the decomposition the eager path built.
    ///
    /// # Errors
    ///
    /// Propagates the device readback failure.
    fn host_factor(&self, device: &WgpuDevice) -> Result<&leto_ops::CholeskyDecomposition<f32>> {
        if let Some(factor) = self.inner.get() {
            return Ok(factor);
        }

        let mut data = vec![0.0f32; self.n * self.n];
        device.download(&self.lower, &mut data)?;
        let factor = leto_ops::CholeskyDecomposition::from_raw_parts(
            leto::Array2::from_shape_vec([self.n, self.n], data)
                .expect("invariant: n*n elements form an n x n factor"),
        );
        // A racing initializer computed the same value from the same
        // immutable buffer, so either winner is correct.
        let _ = self.inner.set(factor);
        Ok(self
            .inner
            .get()
            .expect("invariant: set unconditionally above"))
    }

    /// Matrix dimension *n*.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the lower-triangular factor buffer on the device.
    #[must_use]
    #[inline]
    pub fn lower(&self) -> &WgpuBuffer<f32> {
        &self.lower
    }

    /// Consume and return the lower-triangular factor buffer.
    #[must_use]
    #[inline]
    pub fn into_lower(self) -> WgpuBuffer<f32> {
        self.lower
    }

    /// Determinant det(**A**) = Πᵢ Lᵢᵢ² from the retained factor diagonal.
    ///
    /// Multiplies in ascending `i` from `1.0`, the same order and the same
    /// operations as [`leto_ops::CholeskyDecomposition::det`], so the result
    /// is bitwise equal to the host reference's on an identical factor. No
    /// device dispatch and no readback: the diagonal is captured while the
    /// factorization already has each panel on the host.
    #[must_use]
    #[inline]
    pub fn det(&self) -> f32 {
        self.diagonal
            .iter()
            .fold(1.0f32, |product, &diagonal| product * (diagonal * diagonal))
    }

    /// Solve **A** · **x** = **rhs** on the device against the resident factor.
    ///
    /// Blocked forward substitution on **L** then backward substitution on
    /// **L**ᵀ, blocked by 256 rows; neither the factor nor the right-hand
    /// side crosses the host, and the host factor stays unmaterialized.
    ///
    /// # Errors
    ///
    /// - `LengthMismatch` when `rhs.len != n`.
    /// - `DispatchFailed` when a dimension or workgroup count exceeds `u32`.
    pub fn solve(&self, device: &WgpuDevice, rhs: &WgpuBuffer<f32>) -> Result<WgpuBuffer<f32>> {
        device_solve(device, &self.lower, self.n, rhs)
    }

    /// Whether the host-side factor copy is resident.
    ///
    /// `false` after [`cholesky_decompose_blocked`] until the first
    /// [`inv`](Self::inv); `true` from construction for
    /// [`cholesky_decompose`]. [`solve`](Self::solve) never materializes it.
    #[must_use]
    #[inline]
    pub fn host_factor_materialized(&self) -> bool {
        self.inner.get().is_some()
    }

    /// Compute the inverse **A**⁻¹ via the host-side decomposition.
    ///
    /// On the blocked entry point's first call this materializes the host
    /// factor (see [`GpuCholesky`]); later calls reuse it. The `n` device
    /// solves that would retire this download are their own item.
    ///
    /// # Errors
    ///
    /// The device readback failure, or the host inversion failure.
    pub fn inv(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
        if self.n == 0 {
            return device.alloc_zeroed::<f32>(0);
        }
        let inv = self
            .host_factor(device)?
            .inv()
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("Cholesky inverse failed: {e}"),
            })?;
        device.upload(leto::Storage::as_slice(inv.storage()))
    }
}
// ---------------------------------------------------------------------------
// Entry point 1 — host delegation (unchanged)
// ---------------------------------------------------------------------------

/// Compute the Cholesky factorization **A** = **L** **L**ᵀ on the GPU.
///
/// The entire factorization (panel + trailing) is delegated to the host via
/// [`leto_ops`].  The result is stored on the device for downstream GPU
/// consumers.  For large matrices where the O(n³) trailing update should run
/// on the GPU, prefer [`cholesky_decompose_blocked`].
///
/// # Errors
///
/// - Non-square matrix.
/// - Non-finite values in the input.
/// - Matrix is not positive-definite.
pub fn cholesky_decompose(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuCholesky> {
    let n = validate_square(&matrix)?;
    if n == 0 {
        // An empty factor has an empty diagonal, so `det` yields the empty
        // product `1.0` and `solve`/`inv` short-circuit before reaching
        // `inner` — no host decomposition needs to exist.
        return Ok(GpuCholesky {
            inner: OnceLock::new(),
            diagonal: Vec::new(),
            lower: device.alloc_zeroed::<f32>(0)?,
            n: 0,
        });
    }

    // Download input to host.
    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    // Create a leto ArrayView over the downloaded data.
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);

    // Factor on CPU using leto-ops.
    let chol =
        leto_ops::cholesky_decompose(&view).map_err(|e| HephaestusError::DispatchFailed {
            message: format!("Cholesky decomposition failed: {e}"),
        })?;

    // Upload the lower-triangular factor to the device.
    let factor = leto::Storage::as_slice(chol.lower().storage());
    let diagonal = (0..n).map(|k| factor[k * n + k]).collect();
    let lower = device.upload(factor)?;

    // This path factored on the host, so the decomposition already exists:
    // publish it rather than making `solve`/`inv` download it back.
    let inner = OnceLock::new();
    let _ = inner.set(chol);

    Ok(GpuCholesky {
        inner,
        diagonal,
        lower,
        n,
    })
}
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
/// reason this entry point exists: unlike [`cholesky_decompose`] which
/// delegates *all* O(n³) work to the CPU, this function offloads the
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
