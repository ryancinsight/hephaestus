#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "ratchet HEPH-UNWRAP-1: pre-existing test-only debt"
    )
)]

//! Shared CPU-side panel factorisation routines for blocked decomposition,
//! plus backend-neutral operand validation.
//!
//! The panel routines operate on packed row-major `f32` slices and return the
//! bookkeeping data the blocked loop needs (permutation vectors, Householder
//! heads, β coefficients). The
//! [`validate_square_operand`](crate::domain::decomposition::validate_square_operand) and
//! [`require_dense_operand`](crate::domain::decomposition::require_dense_operand) validators
//! check a rank-2 [`leto::Layout`] once so
//! both backends share the same dispatch preconditions. All are used by both
//! the wgpu and CUDA backends.

pub mod blocked;

pub use blocked::{
    BlockedCholeskyBackend, BlockedCholeskyFactors, BlockedDecompositionBackend, BlockedLuFactors,
    BlockedQrBackend, BlockedQrFactors, PanelRegion, TrailingGemm, TrailingHh, TrailingSyrk,
    blocked_cholesky, blocked_lu, blocked_qr,
};

use crate::domain::error::{HephaestusError, Result};
use crate::domain::planning::map_layout_err;
use leto::Layout;

/// Validate that a rank-2 operand is square and fits its buffer, returning its
/// dimension `n`.
///
/// # Errors
/// Returns [`HephaestusError::DispatchFailed`] when the shape is not square or
/// the layout does not fit `buf_len` elements.
pub fn validate_square_operand(layout: &Layout<2>, buf_len: usize) -> Result<usize> {
    let [rows, cols] = layout.shape();
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("Decomposition requires a square matrix, got shape [{rows}, {cols}]"),
        });
    }
    layout
        .validate_storage_len(buf_len)
        .map_err(map_layout_err)?;
    Ok(rows)
}

/// Require a dense C-contiguous zero-offset operand for a blocked decomposition.
///
/// The blocked entry points bulk-copy `rows·cols` elements straight from the
/// operand's raw buffer. A transposed, offset, or broadcast (zero-stride)
/// layout would make that raw copy compute from the wrong elements — and for
/// layouts whose validated storage extent is smaller than `rows·cols`
/// (broadcast), read past the allocation. `validate_storage_len` only bounds
/// the layout's own extent, so density is checked explicitly here before any
/// raw copy.
///
/// # Errors
/// Returns [`HephaestusError::DispatchFailed`] naming the shape, strides, and
/// offset when the operand is not dense C-contiguous at offset 0.
pub fn require_dense_operand(label: &str, layout: &Layout<2>) -> Result<()> {
    if layout.is_c_contiguous() {
        return Ok(());
    }
    Err(HephaestusError::DispatchFailed {
        message: format!(
            "{label} blocked decomposition requires a dense C-contiguous zero-offset operand; \
             got shape {:?}, strides {:?}, offset {} — materialize the view first (e.g. an \
             identity strided copy into a fresh buffer)",
            layout.shape(),
            layout.strides(),
            layout.offset()
        ),
    })
}

/// Validate that every element of `a` is finite.
///
/// Used by all panel factorisation routines before the main computation loop.
/// Consolidates the identical non-finite-check pattern to a single SSOT.
#[inline]
fn validate_finite(a: &[f32], op: &str) -> Result<()> {
    if let Some((idx, value)) = a.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("{op} failed: entry {idx} is non-finite ({value})"),
        });
    }
    Ok(())
}

/// In-place partial-pivoting LU factorisation of a packed *n* × *n*
/// row-major matrix, returning the LAPACK-style cumulative row
/// permutation vector.
///
/// After factorisation the strictly-lower triangle stores the unit-lower
/// **L** entries (the diagonal of L is implicit 1) and the upper
/// triangle (including the diagonal) stores **U**.
///
/// Returns the pivot vector where `pivots[k]` is the row swapped with
/// row `k` at step *k* (identity if no swap occurred).
///
/// # Errors
///
/// Factor one blocked-LU panel entirely on the host (backend-neutral).
///
/// This is the shared per-panel computation of the blocked LU loop, identical
/// across the wgpu and CUDA backends: factor the `b × b` diagonal block, apply
/// its row swaps to the cumulative permutation and the row panel, solve
/// `L₂₁ = A₂₁ · U₁₁⁻¹` and `U₁₂ = L₁₁⁻¹ · A₁₂` for the `trail` trailing
/// rows/columns, and write the factored diagonal back into the row panel. All
/// device I/O (region gather/scatter, the trailing GEMM) stays in the backend
/// loop around this call.
///
/// Buffers are compact row-major: `col_panel` is `(n-k) × b` (the diagonal
/// block occupies its first `b × b`, the `trail` trailing rows follow),
/// `row_panel` is `b × n`, and `diag` is scratch of at least `b²` elements.
/// `k` is the panel's starting index in the full `n × n` matrix; `trail`
/// is `n - k - b` (zero on the final panel, which runs the solves zero times).
///
/// # Errors
/// Propagates [`panel_lu_packed`]'s failure (non-finite entry or zero pivot).
#[expect(
    clippy::too_many_arguments,
    reason = "the parameters are the operation's independent descriptors; bundling them into a struct renames the same fields without removing a call-site decision"
)]
pub fn factor_lu_panel(
    col_panel: &mut [f32],
    row_panel: &mut [f32],
    diag: &mut [f32],
    k: usize,
    b: usize,
    n: usize,
    trail: usize,
    perm: &mut [usize],
    sign: &mut i8,
) -> Result<()> {
    // Factor the diagonal block A[k..k+b, k..k+b] (its compact copy is the
    // leading b×b of col_panel).
    for i in 0..b {
        for j in 0..b {
            diag[i * b + j] = col_panel[i * b + j];
        }
    }
    let pivots = panel_lu_packed(&mut diag[..b * b], b)?;

    // Apply the panel's row swaps to the cumulative permutation and the row
    // panel (each row has n elements).
    for (i, &pivot) in pivots.iter().enumerate().take(b) {
        if pivot != i {
            perm.swap(k + i, k + pivot);
            *sign = -*sign;
            for j in 0..n {
                row_panel.swap(i * n + j, pivot * n + j);
            }
        }
    }

    // L₂₁ = A₂₁ · U₁₁⁻¹ (forward substitution over the trailing rows).
    for i in 0..trail {
        for j in 0..b {
            let mut s = col_panel[(b + i) * b + j];
            for p in 0..j {
                s -= col_panel[(b + i) * b + p] * diag[p * b + j];
            }
            col_panel[(b + i) * b + j] = s / diag[j * b + j];
        }
    }

    // U₁₂ = L₁₁⁻¹ · A₁₂ (forward substitution over the trailing columns).
    for j in 0..trail {
        for i in 0..b {
            let mut s = row_panel[i * n + (k + b + j)];
            for p in 0..i {
                s -= diag[i * b + p] * row_panel[p * n + (k + b + j)];
            }
            row_panel[i * n + (k + b + j)] = s;
        }
    }

    // Copy the factored diagonal back into the row panel so it is uploaded.
    for i in 0..b {
        for j in 0..b {
            row_panel[i * n + (k + j)] = diag[i * b + j];
        }
    }

    Ok(())
}

/// In-place lower-triangular Cholesky factorisation of a packed `n × n`
/// row-major symmetric positive-definite matrix.
///
/// Reads the lower triangle of `a` and overwrites it with the lower factor
/// **L** such that `A = L Lᵀ`; the strictly-upper triangle is zeroed. Column
/// Cholesky (the same left-looking order as `leto_ops`), so the blocked
/// backends' differential-vs-leto reconstruction stays within tolerance.
///
/// # Errors
/// - `LengthMismatch` when `a.len() != n * n`.
/// - `DispatchFailed` on a non-finite entry or a non-positive pivot (the
///   matrix is not positive-definite).
pub fn panel_cholesky_packed(a: &mut [f32], n: usize) -> Result<()> {
    if a.len() != n * n {
        return Err(HephaestusError::LengthMismatch {
            host_len: n * n,
            device_len: a.len(),
        });
    }
    validate_finite(a, "Cholesky panel factorisation")?;

    for j in 0..n {
        // Diagonal: L[j,j] = sqrt(A[j,j] - Σ_{p<j} L[j,p]²).
        let mut d = a[j * n + j];
        for p in 0..j {
            d -= a[j * n + p] * a[j * n + p];
        }
        // Inputs are validated finite above, so `d` is finite; a non-positive
        // pivot means the matrix is not positive-definite.
        if d <= 0.0 {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "Cholesky panel factorisation failed: pivot {j} is not positive ({d}); \
                     matrix is not positive-definite"
                ),
            });
        }
        let ljj = d.sqrt();
        a[j * n + j] = ljj;

        // Below the diagonal: L[i,j] = (A[i,j] - Σ_{p<j} L[i,p]·L[j,p]) / L[j,j].
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for p in 0..j {
                s -= a[i * n + p] * a[j * n + p];
            }
            a[i * n + j] = s / ljj;
        }

        // Zero the strictly-upper triangle of this row.
        for c in (j + 1)..n {
            a[j * n + c] = 0.0;
        }
    }

    Ok(())
}

/// Factor one blocked-Cholesky panel entirely on the host (backend-neutral).
///
/// The shared per-panel computation of the blocked Cholesky loop, identical
/// across both backends: factor the `b × b` diagonal block via
/// [`panel_cholesky_packed`] and solve `L₂₁ = A₂₁ · L₁₁⁻ᵀ` for the `trail_rows`
/// trailing rows. `panel` is the compact row-major active panel, its leading
/// `b × b` the diagonal block and the following `trail_rows × b` the
/// off-diagonal rows; both are overwritten with their factored values. Device
/// I/O (region gather/scatter, the SYRK trailing update) stays in the backend
/// loop. On the final panel `trail_rows` is zero and the solve runs zero times.
///
/// # Errors
/// Propagates [`panel_cholesky_packed`]'s failure (non-finite entry or
/// non-positive-definite diagonal block).
pub fn factor_cholesky_panel(panel: &mut [f32], b: usize, trail_rows: usize) -> Result<()> {
    let (diag_part, rhs_part) = panel.split_at_mut(b * b);
    panel_cholesky_packed(diag_part, b)?;

    // L₂₁ = A₂₁ · L₁₁⁻ᵀ: forward-substitute each column against L₁₁.
    for col in 0..b {
        for i in 0..trail_rows {
            let mut s = rhs_part[i * b + col];
            for p in 0..col {
                s -= rhs_part[i * b + p] * diag_part[col * b + p];
            }
            rhs_part[i * b + col] = s / diag_part[col * b + col];
        }
    }

    Ok(())
}

/// - `LengthMismatch` when `a.len() != n * n`.
/// - `DispatchFailed` on non-finite entries or an exact-zero pivot.
pub fn panel_lu_packed(a: &mut [f32], n: usize) -> Result<Vec<usize>> {
    if a.len() != n * n {
        return Err(HephaestusError::LengthMismatch {
            host_len: n * n,
            device_len: a.len(),
        });
    }
    validate_finite(a, "LU panel factorisation")?;

    let mut pivots = vec![0; n];

    for k in 0..n {
        // Partial pivot: find the row r ≥ k with the largest |a[r, k]|.
        let mut pivot_row = k;
        let mut pivot_mag = a[k * n + k].abs();
        for r in (k + 1)..n {
            let mag = a[r * n + k].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = r;
            }
        }
        if pivot_row != k {
            // Swap entire rows in the working portion.
            for c in 0..n {
                a.swap(k * n + c, pivot_row * n + c);
            }
        }
        pivots[k] = pivot_row;

        if pivot_mag == 0.0 {
            return Err(HephaestusError::DispatchFailed {
                message: format!("LU panel factorisation failed: pivot column {k} is exactly zero"),
            });
        }

        let pivot = a[k * n + k];
        for r in (k + 1)..n {
            let factor = a[r * n + k] / pivot;
            a[r * n + k] = factor; // L entry
            for c in (k + 1)..n {
                a[r * n + c] -= factor * a[k * n + c];
            }
        }
    }

    Ok(pivots)
}

/// In-place Householder QR factorisation of an *m* × *n* packed
/// row-major matrix (*m* ≥ *n*), returning the Householder vector
/// heads and β coefficients.
///
/// After factorisation the upper triangle of `a` (including diagonal)
/// stores **R**, and the strictly-lower triangle stores the Householder
/// vector tails.  The heads are stored in the returned `heads` vector
/// because the diagonal slots are occupied by R's diagonal (α).
///
/// Returns `(heads, betas)` where `heads[k]` is the leading entry of
/// the *k*-th Householder vector and `betas[k] = 2 / (vᵀv)`.
///
/// # Errors
///
/// - `LengthMismatch` when `a.len() != m * n`.
/// - `DispatchFailed` when `m < n`, on non-finite entries, or on a
///   zero-norm column (rank-deficient input).
pub fn panel_qr_packed(a: &mut [f32], m: usize, n: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    if a.len() != m * n {
        return Err(HephaestusError::LengthMismatch {
            host_len: m * n,
            device_len: a.len(),
        });
    }
    if m < n {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR panel requires m ≥ n, got [{m}, {n}]"),
        });
    }

    // Validate non-finite inputs via the shared helper.
    validate_finite(a, "QR panel factorisation")?;

    let mut heads = vec![0.0f32; n];
    let mut betas = vec![0.0f32; n];

    for k in 0..n {
        // ‖x‖ for the column segment a[k..m, k].
        let mut norm_sq = 0.0f32;
        for r in k..m {
            let x = a[r * n + k];
            norm_sq += x * x;
        }
        let norm = norm_sq.sqrt();
        if norm == 0.0 {
            return Err(HephaestusError::DispatchFailed {
                message: format!("QR pivot column {k} has zero norm: matrix is rank-deficient"),
            });
        }

        // α = −sign(a[k,k]) · ‖x‖ for cancellation-free head computation.
        let pivot = a[k * n + k];
        let alpha = if pivot > 0.0 { -norm } else { norm };
        let head = pivot - alpha;

        // β = 2 / (vᵀv)  where v = (head, a[k+1,k], ..., a[m-1,k]).
        let mut v_norm_sq = head * head;
        for r in (k + 1)..m {
            let x = a[r * n + k];
            v_norm_sq += x * x;
        }
        let beta = 2.0 / v_norm_sq;

        // Apply H = I − β·v·vᵀ to trailing columns k+1..n-1.
        for c in (k + 1)..n {
            let mut s = head * a[k * n + c];
            for r in (k + 1)..m {
                s += a[r * n + k] * a[r * n + c];
            }
            let bs = beta * s;
            a[k * n + c] -= bs * head;
            for r in (k + 1)..m {
                a[r * n + c] -= bs * a[r * n + k];
            }
        }

        a[k * n + k] = alpha; // R diagonal; v tails remain below.
        heads[k] = head;
        betas[k] = beta;
    }

    Ok((heads, betas))
}

/// Apply a factored packed QR panel's Householder reflectors to a compact
/// row-major trailing matrix in place.
///
/// `panel` is the `panel_rows × panel_cols` output of [`panel_qr_packed`].
/// `heads` and `betas` are the corresponding returned reflector metadata, and
/// `trailing` is a `panel_rows × trailing_cols` compact matrix. Reflectors are
/// applied in increasing column order, computing
/// `C ← (I − β v vᵀ) C` in native `f32` precision.
///
/// This operation does not allocate. It preserves the packed panel and updates
/// only `trailing`.
///
/// # Examples
///
/// ```
/// use hephaestus_core::{apply_packed_qr_panel_left, panel_qr_packed};
///
/// let mut panel = [2.0_f32, 0.0];
/// let (heads, betas) = panel_qr_packed(&mut panel, 2, 1)?;
/// let mut trailing = [1.0_f32, 3.0];
/// apply_packed_qr_panel_left(&panel, 2, 1, &heads, &betas, &mut trailing, 1)?;
/// assert_eq!(trailing, [-1.0, 3.0]);
/// # Ok::<(), hephaestus_core::HephaestusError>(())
/// ```
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when a dimension product
/// overflows, `panel_rows < panel_cols`, or any slice length does not match its
/// declared shape. Returns [`HephaestusError::DispatchFailed`] when reflector
/// metadata or trailing input contains a non-finite value.
pub fn apply_packed_qr_panel_left(
    panel: &[f32],
    panel_rows: usize,
    panel_cols: usize,
    heads: &[f32],
    betas: &[f32],
    trailing: &mut [f32],
    trailing_cols: usize,
) -> Result<()> {
    let panel_len = panel_rows.checked_mul(panel_cols).ok_or_else(|| {
        HephaestusError::InvalidConfiguration {
            message: format!("QR panel shape [{panel_rows}, {panel_cols}] overflows element count"),
        }
    })?;
    let trailing_len = panel_rows.checked_mul(trailing_cols).ok_or_else(|| {
        HephaestusError::InvalidConfiguration {
            message: format!(
                "QR trailing shape [{panel_rows}, {trailing_cols}] overflows element count"
            ),
        }
    })?;
    if panel_rows < panel_cols {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "QR packed-panel application requires rows ≥ columns, got \
                 [{panel_rows}, {panel_cols}]"
            ),
        });
    }
    for (name, actual, expected) in [
        ("panel", panel.len(), panel_len),
        ("heads", heads.len(), panel_cols),
        ("betas", betas.len(), panel_cols),
        ("trailing", trailing.len(), trailing_len),
    ] {
        if actual != expected {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "QR packed-panel {name} length {actual} does not match expected {expected}"
                ),
            });
        }
    }

    validate_finite(panel, "QR packed panel")?;
    validate_finite(heads, "QR packed-panel heads")?;
    validate_finite(betas, "QR packed-panel betas")?;
    validate_finite(trailing, "QR packed-panel trailing application")?;

    for reflector in 0..panel_cols {
        let head = heads[reflector];
        let beta = betas[reflector];
        for col in 0..trailing_cols {
            let mut dot = head * trailing[reflector * trailing_cols + col];
            for row in (reflector + 1)..panel_rows {
                dot += panel[row * panel_cols + reflector] * trailing[row * trailing_cols + col];
            }
            let scale = beta * dot;
            trailing[reflector * trailing_cols + col] -= scale * head;
            for row in (reflector + 1)..panel_rows {
                trailing[row * trailing_cols + col] -= scale * panel[row * panel_cols + reflector];
            }
        }
    }

    Ok(())
}

/// Split a packed LU factorisation into explicit dense **L** and **U**
/// matrices.
///
/// `packed` is the in-place result of a packed LU factorisation (e.g.
/// [`panel_lu_packed`]) of an *n* × *n* row-major matrix: the
/// strictly-lower triangle stores the unit-lower **L** entries (the
/// diagonal of L is implicit 1) and the upper triangle (including the
/// diagonal) stores **U**.
///
/// Returns `(l, u)` as dense row-major *n* × *n* matrices where `l`
/// carries an explicit unit diagonal and zeros above, and `u` carries
/// zeros below the diagonal, so `l · u` reproduces the packed factor
/// product. The split is a pure copy: every output entry equals the
/// corresponding packed entry (or the structural constant 0/1), so no
/// rounding occurs.
///
/// # Errors
///
/// - `LengthMismatch` when `packed.len() != n * n`.
pub fn split_packed_lu(packed: &[f32], n: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    if packed.len() != n * n {
        return Err(HephaestusError::LengthMismatch {
            host_len: n * n,
            device_len: packed.len(),
        });
    }
    let mut l = vec![0.0f32; n * n];
    let mut u = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            let idx = r * n + c;
            let val = packed[idx];
            match r.cmp(&c) {
                core::cmp::Ordering::Greater => l[idx] = val,
                core::cmp::Ordering::Equal => {
                    l[idx] = 1.0;
                    u[idx] = val;
                }
                core::cmp::Ordering::Less => u[idx] = val,
            }
        }
    }
    Ok((l, u))
}

#[cfg(test)]
#[path = "tests_decomposition.rs"]
mod tests;
