//! Shared CPU-side panel factorisation routines for blocked decomposition.
//!
//! These pure-arithmetic helpers are used by both the wgpu and CUDA
//! backends.  They operate on packed row-major `f32` slices and return
//! the bookkeeping data the blocked loop needs (permutation vectors,
//! Householder heads, β coefficients).

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
/// - `LengthMismatch` when `a.len() != n * n`.
/// - `DispatchFailed` on non-finite entries or an exact-zero pivot.
pub fn panel_lu_packed(a: &mut [f32], n: usize) -> Result<Vec<usize>> {
    if a.len() != n * n {
        return Err(HephaestusError::LengthMismatch {
            host_len: n * n,
            device_len: a.len(),
        });
    }
    if let Some((idx, value)) = a
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(HephaestusError::DispatchFailed {
            message: format!("LU panel factorisation failed: entry {idx} is non-finite ({value})"),
        });
    }

    let mut pivots: Vec<usize> = (0..n).collect();

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
            pivots.swap(k, pivot_row);
        }

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

    // Validate non-finite inputs.
    if let Some((idx, value)) = a
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR panel factorisation failed: entry {idx} is non-finite ({value})"),
        });
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::error::HephaestusError;

    // ── panel_lu_packed ──────────────────────────────────────────────

    /// Reconstruct the original matrix **A** from the packed LU result and
    /// the pivot vector.
    ///
    /// The packed result satisfies P·A = L·U, where P is the cumulative row
    /// permutation recorded in `pivots`.  This helper extracts L and U,
    /// computes L·U, then applies the inverse permutation to recover A.
    fn lu_reconstruct_original(a_factored: &[f32], pivots: &[usize], n: usize) -> Vec<f32> {
        // Build L (unit lower) and U (upper) from the packed result.
        let mut l = vec![0.0f32; n * n];
        let mut u = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    l[i * n + j] = a_factored[i * n + j];
                } else {
                    l[i * n + j] = if i == j { 1.0 } else { 0.0 };
                    u[i * n + j] = a_factored[i * n + j];
                }
            }
        }
        // Compute L·U.
        let mut lu = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0f32;
                for k in 0..n {
                    s += l[i * n + k] * u[k * n + j];
                }
                lu[i * n + j] = s;
            }
        }
        // Apply inverse permutation: P·A = L·U  =>  A[i,:] = (P·A)[inv_perm[i],:].
        // pivots[i] = original row at factored position i.
        // inv_perm[j] = factored position of original row j.
        let mut inv_perm = vec![0usize; n];
        for i in 0..n {
            inv_perm[pivots[i]] = i;
        }
        let mut a = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = lu[inv_perm[i] * n + j];
            }
        }
        a
    }

    #[test]
    fn lu_identity_matrix_yields_identity_factors() {
        let mut a = vec![1.0f32, 0.0, 0.0, 1.0];
        let pivots = panel_lu_packed(&mut a, 2).unwrap();
        // No swaps should have occurred.
        assert_eq!(pivots, vec![0, 1]);
        // L should be identity (no below-diagonal entries).
        assert_eq!(a[2], 0.0); // L[1,0]
                               // U should be identity.
        assert_eq!(a[0], 1.0); // U[0,0]
        assert_eq!(a[3], 1.0); // U[1,1]
    }

    #[test]
    fn lu_known_2x2_system_reconstructs_correctly() {
        // A = [[2, 1], [4, 3]]
        let original = vec![2.0f32, 1.0, 4.0, 3.0];
        let mut a = original.clone();
        let pivots = panel_lu_packed(&mut a, 2).unwrap();

        // Reconstruct A from P·A = L·U and verify against original.
        let recovered = lu_reconstruct_original(&a, &pivots, 2);
        for i in 0..4 {
            assert!(
                (recovered[i] - original[i]).abs() <= 1e-6,
                "LU reconstruction mismatch at {i}: got {}, expected {}",
                recovered[i],
                original[i]
            );
        }
    }

    #[test]
    fn lu_known_3x3_system_with_pivoting() {
        // A = [[0, 2, 1], [1, 0, 3], [4, 1, 0]]
        // This requires row swaps since a[0,0] = 0.
        let original = vec![0.0f32, 2.0, 1.0, 1.0, 0.0, 3.0, 4.0, 1.0, 0.0];
        let mut a = original.clone();
        let pivots = panel_lu_packed(&mut a, 3).unwrap();

        // A swap must have occurred at step 0 (pivot column 0 is zero).
        assert_ne!(pivots[0], 0, "must pivot at step 0 since a[0,0]=0");

        // Reconstruct original A from P·A = L·U.
        let recovered = lu_reconstruct_original(&a, &pivots, 3);
        for i in 0..9 {
            assert!(
                (recovered[i] - original[i]).abs() <= 1e-5,
                "LU reconstruction mismatch at {i}: got {}, expected {}",
                recovered[i],
                original[i]
            );
        }
    }

    #[test]
    fn lu_larger_5x5_system_reconstructs_correctly() {
        // Well-conditioned 5×5 matrix.
        #[rustfmt::skip]
        let original = vec![
            10.0, 1.0, 0.5, 0.2, 0.1,
             1.0, 8.0, 0.3, 0.1, 0.05,
             0.5, 0.3, 6.0, 0.2, 0.1,
             0.2, 0.1, 0.2, 4.0, 0.3,
             0.1, 0.05, 0.1, 0.3, 2.0,
        ];
        let mut a = original.clone();
        let pivots = panel_lu_packed(&mut a, 5).unwrap();

        let recovered = lu_reconstruct_original(&a, &pivots, 5);
        for i in 0..25 {
            assert!(
                (recovered[i] - original[i]).abs() <= 1e-4,
                "5×5 LU reconstruction mismatch at {i}: got {}, expected {}",
                recovered[i],
                original[i]
            );
        }
    }

    #[test]
    fn lu_singular_matrix_is_rejected() {
        // Singular: [[0, 0], [0, 1]]
        let mut a = vec![0.0f32, 0.0, 0.0, 1.0];
        let result = panel_lu_packed(&mut a, 2);
        assert!(result.is_err(), "singular matrix must be rejected");
    }

    #[test]
    fn lu_non_finite_entry_is_rejected() {
        let mut a = vec![1.0f32, f32::NAN, 0.0, 1.0];
        let result = panel_lu_packed(&mut a, 2);
        assert!(matches!(
            result,
            Err(HephaestusError::DispatchFailed { message })
                if message.contains("non-finite")
        ));
    }

    #[test]
    fn lu_length_mismatch_is_rejected() {
        let mut a = vec![1.0f32, 0.0, 0.0]; // 3 elements, but n=2 requires 4
        let result = panel_lu_packed(&mut a, 2);
        assert!(matches!(
            result,
            Err(HephaestusError::LengthMismatch {
                host_len: 4,
                device_len: 3
            })
        ));
    }

    #[test]
    fn lu_pivots_track_cumulative_permutation() {
        // A = [[0, 0, 1], [1, 0, 0], [0, 1, 0]] — requires two swaps.
        let original = vec![0.0f32, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut a = original.clone();
        let pivots = panel_lu_packed(&mut a, 3).unwrap();

        // Verify the permutation is valid (each value 0..n appears once).
        let mut sorted = pivots.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);

        // Verify reconstruction.
        let recovered = lu_reconstruct_original(&a, &pivots, 3);
        for i in 0..9 {
            assert!(
                (recovered[i] - original[i]).abs() <= 1e-5,
                "permuted LU reconstruction mismatch at {i}"
            );
        }
    }

    // ── panel_qr_packed ──────────────────────────────────────────────

    /// Reconstruct A from the packed Householder QR result and verify
    /// that the upper triangle of the packed result matches the expected R.
    fn qr_reconstruct_r(a_packed: &[f32], m: usize, n: usize) -> Vec<f32> {
        let mut r = vec![0.0f32; m * n];
        for i in 0..n {
            for j in i..n {
                r[i * n + j] = a_packed[i * n + j];
            }
        }
        r
    }

    /// Apply the Householder reflectors from the packed QR result to the
    /// identity matrix to reconstruct Q, then verify Q^T Q = I.
    fn qr_reconstruct_q(
        a_packed: &[f32],
        heads: &[f32],
        betas: &[f32],
        m: usize,
        n: usize,
    ) -> Vec<f32> {
        // Start with Q = I (m×m).
        let mut q = vec![0.0f32; m * m];
        for i in 0..m {
            q[i * m + i] = 1.0;
        }

        // Apply Hₖ = I − βₖ vₖ vₖᵀ for k = n-1, ..., 0 (reverse order).
        for k in (0..n).rev() {
            let vec_len = m - k;
            // Reconstruct vₖ.
            let mut v = vec![0.0f32; vec_len];
            v[0] = heads[k];
            for i in 1..vec_len {
                v[i] = a_packed[(k + i) * n + k];
            }
            let beta = betas[k];

            // Apply Hₖ to each column of Q: Q[:, j] -= β · v · (vᵀ · Q[k:m, j])
            for j in 0..m {
                let mut dot = 0.0f32;
                for i in 0..vec_len {
                    dot += v[i] * q[(k + i) * m + j];
                }
                for i in 0..vec_len {
                    q[(k + i) * m + j] -= beta * v[i] * dot;
                }
            }
        }
        q
    }

    #[test]
    fn qr_identity_2x2_yields_correct_r() {
        let original = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut a = original.clone();
        let (heads, betas) = panel_qr_packed(&mut a, 2, 2).unwrap();

        // Panel QR stores α = −sign(a[k,k])·‖x‖ on the diagonal, so R
        // diagonal entries can be negative.  Check |R[i,i]| = ‖col i‖.
        assert!((a[0].abs() - 1.0).abs() <= 1e-6); // |R[0,0]| = 1
        assert!(a[1].abs() <= 1e-6); // R[0,1] = 0
        assert!((a[3].abs() - 1.0).abs() <= 1e-6); // |R[1,1]| = 1

        // Reconstruct Q and verify Q·R ≈ A.
        let q = qr_reconstruct_q(&a, &heads, &betas, 2, 2);
        for i in 0..2 {
            for j in 0..2 {
                let mut qr_val = 0.0f32;
                for k in 0..2 {
                    qr_val += q[i * 2 + k] * a[k * 2 + j]; // R is in upper triangle of a
                }
                assert!(
                    (qr_val - original[i * 2 + j]).abs() <= 1e-5,
                    "Q·R[{i},{j}] = {qr_val}, expected {}",
                    original[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn qr_known_3x2_system_has_upper_triangular_r() {
        // A = [[1, 0], [0, 1], [1, 1]]
        let original = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut a = original.clone();
        let (heads, betas) = panel_qr_packed(&mut a, 3, 2).unwrap();

        // R must be upper triangular: a[1,0] should be zero (below diagonal).
        assert!(a[2].abs() <= 1e-5, "R[1,0] should be zero, got {}", a[2]);

        // Reconstruct Q and verify Q^T Q = I.
        let q = qr_reconstruct_q(&a, &heads, &betas, 3, 2);
        // Q is 3×3, verify Q^T Q = I₃.
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0f32;
                for k in 0..3 {
                    dot += q[k * 3 + i] * q[k * 3 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() <= 1e-4,
                    "Q^T Q[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }

        // Verify Q·R ≈ A.
        let r = qr_reconstruct_r(&a, 3, 2);
        for i in 0..3 {
            for j in 0..2 {
                let mut qr_val = 0.0f32;
                for k in 0..3 {
                    qr_val += q[i * 3 + k] * r[k * 2 + j];
                }
                assert!(
                    (qr_val - original[i * 2 + j]).abs() <= 1e-4,
                    "Q·R[{i},{j}] = {qr_val}, expected {}",
                    original[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn qr_5x3_system_orthogonality_and_reconstruction() {
        // Well-conditioned 5×3 matrix.
        #[rustfmt::skip]
        let original = vec![
            3.0, 1.0, 0.5,
            1.0, 4.0, 0.3,
            0.5, 0.3, 5.0,
            0.2, 0.1, 0.2,
            0.1, 0.05, 0.1,
        ];
        let mut a = original.clone();
        let (heads, betas) = panel_qr_packed(&mut a, 5, 3).unwrap();

        // NOTE: In the packed format, strictly below-diagonal entries store
        // Householder vector tails, NOT zeros.  Only the upper triangle
        // (j >= i) contains R entries.

        // Reconstruct Q and verify Q^T Q = I₅.
        let q = qr_reconstruct_q(&a, &heads, &betas, 5, 3);
        for i in 0..5 {
            for j in 0..5 {
                let mut dot = 0.0f32;
                for k in 0..5 {
                    dot += q[k * 5 + i] * q[k * 5 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() <= 1e-4,
                    "Q^T Q[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }

        // Verify Q·R ≈ A using the packed matrix directly.
        // R[i,j] = a[i*n+j] for j >= i, and 0 for j < i.
        for i in 0..5 {
            for j in 0..3 {
                let mut qr_val = 0.0f32;
                for k in 0..5.min(3) {
                    let r_kj = if j >= k { a[k * 3 + j] } else { 0.0 };
                    qr_val += q[i * 5 + k] * r_kj;
                }
                assert!(
                    (qr_val - original[i * 3 + j]).abs() <= 1e-3,
                    "Q·R[{i},{j}] = {qr_val}, expected {}",
                    original[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn qr_diagonal_matrix_has_known_r() {
        // Diagonal 3×3: R should equal ±the diagonal.
        let original = vec![2.0f32, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        let mut a = original.clone();
        let (heads, betas) = panel_qr_packed(&mut a, 3, 3).unwrap();

        // |R[i,i]| should equal the column norms (diagonal entries).
        assert!((a[0].abs() - 2.0).abs() <= 1e-5);
        assert!((a[4].abs() - 3.0).abs() <= 1e-5);
        assert!((a[8].abs() - 4.0).abs() <= 1e-5);
        // Off-diagonal R entries (upper triangle) should be zero.
        assert!(a[1].abs() <= 1e-5);
        assert!(a[2].abs() <= 1e-5);
        assert!(a[5].abs() <= 1e-5);

        // Verify Q·R ≈ A for the diagonal case.
        let q = qr_reconstruct_q(&a, &heads, &betas, 3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut qr_val = 0.0f32;
                for k in 0..3 {
                    qr_val += q[i * 3 + k] * a[k * 3 + j];
                }
                assert!(
                    (qr_val - original[i * 3 + j]).abs() <= 1e-3,
                    "Q·R[{i},{j}] = {qr_val}, expected {}",
                    original[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn qr_rank_deficient_column_is_rejected() {
        // Column 1 is all zeros: [[1, 0], [0, 0]]
        let mut a = vec![1.0f32, 0.0, 0.0, 0.0];
        let result = panel_qr_packed(&mut a, 2, 2);
        assert!(matches!(
            result,
            Err(HephaestusError::DispatchFailed { message })
                if message.contains("rank-deficient")
        ));
    }

    #[test]
    fn qr_underdetermined_is_rejected() {
        // m < n: 2×3 matrix
        let mut a = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let result = panel_qr_packed(&mut a, 2, 3);
        assert!(matches!(
            result,
            Err(HephaestusError::DispatchFailed { message })
                if message.contains("m ≥ n")
        ));
    }

    #[test]
    fn qr_non_finite_entry_is_rejected() {
        let mut a = vec![1.0f32, f32::INFINITY, 0.0, 1.0];
        let result = panel_qr_packed(&mut a, 2, 2);
        assert!(matches!(
            result,
            Err(HephaestusError::DispatchFailed { message })
                if message.contains("non-finite")
        ));
    }

    #[test]
    fn qr_length_mismatch_is_rejected() {
        let mut a = vec![1.0f32, 0.0, 0.0]; // 3 elements, but 2×2 requires 4
        let result = panel_qr_packed(&mut a, 2, 2);
        assert!(matches!(
            result,
            Err(HephaestusError::LengthMismatch {
                host_len: 4,
                device_len: 3
            })
        ));
    }

    #[test]
    fn qr_betas_are_positive() {
        let mut a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (_, betas) = panel_qr_packed(&mut a, 3, 2).unwrap();
        for (k, &beta) in betas.iter().enumerate() {
            // β = 2/(vᵀv) is always positive.  No upper bound is
            // guaranteed: β can exceed 2 when ‖v‖ < 1.
            assert!(beta > 0.0, "beta[{k}] = {beta} should be positive");
        }
    }
}
