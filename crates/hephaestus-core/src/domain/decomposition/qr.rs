use super::validate::validate_finite;
use crate::domain::error::{HephaestusError, Result};

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
