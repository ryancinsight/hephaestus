use super::validate::validate_finite;
use crate::domain::error::{HephaestusError, Result};

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
