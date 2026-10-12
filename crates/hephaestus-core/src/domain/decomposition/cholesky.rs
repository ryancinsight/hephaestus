use super::validate::validate_finite;
use crate::domain::error::{HephaestusError, Result};

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
