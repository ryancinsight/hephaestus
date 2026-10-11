use super::*;
use crate::domain::error::HephaestusError;
use crate::test_support::assert_rejects;

// ── operand validators ───────────────────────────────────────────

#[test]
fn validate_square_operand_accepts_square_and_rejects_rectangular() {
    let square = Layout::c_contiguous([3, 3]).expect("contiguous");
    assert_eq!(validate_square_operand(&square, 9).expect("square"), 3);

    let rect = Layout::c_contiguous([2, 3]).expect("contiguous");
    assert!(matches!(
        validate_square_operand(&rect, 6).unwrap_err(),
        HephaestusError::DispatchFailed { .. }
    ));

    // Layout larger than its buffer is rejected.
    assert!(validate_square_operand(&square, 8).is_err());
}

#[test]
fn panel_cholesky_packed_reconstructs_spd_matrix() {
    // SPD matrix A = MᵀM for a well-conditioned M; factor and check LLᵀ = A.
    let n = 3;
    let a = [4.0f32, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
    let mut l = a;
    panel_cholesky_packed(&mut l, n).expect("spd");
    // Expected L (classic textbook result): [[2,0,0],[6,1,0],[-8,5,3]].
    let expected_l = [2.0f32, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0];
    for (got, want) in l.iter().zip(expected_l.iter()) {
        assert!((got - want).abs() <= 1e-4, "L: got {got} want {want}");
    }
    // Reconstruct A = L Lᵀ and compare.
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..n {
                s += l[i * n + p] * l[j * n + p];
            }
            assert!(
                (s - a[i * n + j]).abs() <= 1e-3,
                "A[{i},{j}]: {s} vs {}",
                a[i * n + j]
            );
        }
    }
}

#[test]
fn panel_cholesky_packed_rejects_non_positive_definite() {
    // Indefinite matrix (negative eigenvalue) must be rejected.
    let mut a = [1.0f32, 2.0, 2.0, 1.0];
    assert!(matches!(
        panel_cholesky_packed(&mut a, 2).unwrap_err(),
        HephaestusError::DispatchFailed { .. }
    ));
}

#[test]
fn factor_lu_panel_single_panel_matches_panel_lu() {
    // One panel covering the whole 3×3 matrix (trail = 0): factor_lu_panel
    // must produce the same packed factors (with pivot rows applied) as a
    // direct panel_lu_packed, and the same cumulative permutation.
    let a = [4.0f32, 3.0, 2.0, 2.0, 1.0, 3.0, 1.0, 5.0, 1.0];
    let n = 3;
    let b = 3;

    // Reference: direct packed LU + its pivot vector applied to a fresh perm.
    let mut ref_packed = a;
    let ref_pivots = panel_lu_packed(&mut ref_packed, n).expect("ref lu");
    let mut ref_perm: Vec<usize> = (0..n).collect();
    for (i, &p) in ref_pivots.iter().enumerate() {
        if p != i {
            ref_perm.swap(i, p);
        }
    }

    // factor_lu_panel: col_panel = row_panel = the full matrix (n×n).
    let mut col_panel = a.to_vec();
    let mut row_panel = a.to_vec();
    let mut diag = vec![0.0f32; b * b];
    let mut perm: Vec<usize> = (0..n).collect();
    let mut sign = 1i8;
    factor_lu_panel(
        &mut col_panel,
        &mut row_panel,
        &mut diag,
        0,
        b,
        n,
        0,
        &mut perm,
        &mut sign,
    )
    .expect("panel");

    // row_panel now holds the pivot-applied packed factors == ref_packed
    // reordered by the same pivots (panel_lu_packed already applied them in
    // ref_packed), so row_panel equals ref_packed exactly.
    for (got, want) in row_panel.iter().zip(ref_packed.iter()) {
        assert!((got - want).abs() <= 1e-6, "got {got} want {want}");
    }
    assert_eq!(perm, ref_perm);
}

#[test]
fn require_dense_operand_rejects_non_contiguous_views() {
    let dense = Layout::c_contiguous([4, 4]).expect("contiguous");
    assert!(require_dense_operand("lu", &dense).is_ok());

    let transposed = Layout::try_new([4, 4], [1, 4], 0).expect("valid test layout");
    assert!(require_dense_operand("lu", &transposed).is_err());

    let offset = Layout::try_new([2, 2], [2, 1], 3).expect("valid test layout");
    assert!(require_dense_operand("qr", &offset).is_err());

    let broadcast = Layout::try_new([4, 4], [0, 1], 0).expect("valid test layout");
    assert!(require_dense_operand("cholesky", &broadcast).is_err());
}

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
    // Apply transpositions in reverse order to recover A from L·U (A = Pᵀ · L · U).
    let mut a = lu;
    for k in (0..n).rev() {
        let pivot = pivots[k];
        if pivot != k {
            for j in 0..n {
                a.swap(k * n + j, pivot * n + j);
            }
        }
    }
    a
}

#[test]
fn lu_identity_matrix_yields_identity_factors() {
    let mut a = vec![1.0f32, 0.0, 0.0, 1.0];
    let pivots = panel_lu_packed(&mut a, 2)
        .expect("invariant: well-formed 2x2 test input factorises without error");
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
    let pivots = panel_lu_packed(&mut a, 2)
        .expect("invariant: well-formed 2x2 test input factorises without error");

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
    let pivots = panel_lu_packed(&mut a, 3)
        .expect("invariant: well-formed 3x3 test input factorises without error");

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
    let pivots = panel_lu_packed(&mut a, 5)
        .expect("invariant: well-formed 5x5 test input factorises without error");

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
    assert_rejects(
        &result,
        "kernel dispatch failed: LU panel factorisation failed: pivot column 0 is exactly zero",
    );
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
    let pivots = panel_lu_packed(&mut a, 3)
        .expect("invariant: well-formed 3x3 test input factorises without error");

    // Since pivots is a sequence of transpositions, we construct the cumulative
    // permutation vector to verify it is a valid permutation of 0..n.
    let mut perm: Vec<usize> = (0..3).collect();
    for (i, &pivot) in pivots.iter().enumerate() {
        perm.swap(i, pivot);
    }
    let mut sorted = perm.clone();
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
    let (heads, betas) = panel_qr_packed(&mut a, 2, 2)
        .expect("invariant: well-formed 2x2 test input factorises without error");

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
    let (heads, betas) = panel_qr_packed(&mut a, 3, 2)
        .expect("invariant: well-formed 3x2 test input factorises without error");

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
    let (heads, betas) = panel_qr_packed(&mut a, 5, 3)
        .expect("invariant: well-formed 5x3 test input factorises without error");

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
            for k in 0..3 {
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
    let (heads, betas) = panel_qr_packed(&mut a, 3, 3)
        .expect("invariant: well-formed 3x3 test input factorises without error");

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
    let (_, betas) = panel_qr_packed(&mut a, 3, 2)
        .expect("invariant: well-formed 3x2 test input factorises without error");
    for (k, &beta) in betas.iter().enumerate() {
        // β = 2/(vᵀv) is always positive.  No upper bound is
        // guaranteed: β can exceed 2 when ‖v‖ < 1.
        assert!(beta > 0.0, "beta[{k}] = {beta} should be positive");
    }
}

#[test]
fn packed_qr_panel_application_matches_full_factorization() {
    let (rows, panel_cols, trailing_cols) = (5usize, 2usize, 1usize);
    let cols = panel_cols + trailing_cols;
    let original = [
        4.0f32, 1.0, 2.0, 1.0, 5.0, 3.0, 0.5, 1.5, 6.0, 0.25, 0.75, 1.25, 0.1, 0.2, 0.4,
    ];

    let mut expected = original;
    let (expected_heads, expected_betas) =
        panel_qr_packed(&mut expected, rows, cols).expect("full QR factorization");

    let mut panel = vec![0.0; rows * panel_cols];
    let mut trailing = vec![0.0; rows * trailing_cols];
    for row in 0..rows {
        panel[row * panel_cols..(row + 1) * panel_cols]
            .copy_from_slice(&original[row * cols..row * cols + panel_cols]);
        trailing[row] = original[row * cols + panel_cols];
    }
    let (heads, betas) =
        panel_qr_packed(&mut panel, rows, panel_cols).expect("first panel factorization");
    apply_packed_qr_panel_left(
        &panel,
        rows,
        panel_cols,
        &heads,
        &betas,
        &mut trailing,
        trailing_cols,
    )
    .expect("packed panel application");
    let (tail_heads, tail_betas) = panel_qr_packed(
        &mut trailing[panel_cols * trailing_cols..],
        rows - panel_cols,
        trailing_cols,
    )
    .expect("tail factorization");

    for row in 0..rows {
        for col in 0..cols {
            let actual = if col < panel_cols {
                panel[row * panel_cols + col]
            } else {
                trailing[row]
            };
            assert_eq!(
                actual,
                expected[row * cols + col],
                "packed QR value [{row},{col}]"
            );
        }
    }
    assert_eq!(heads, expected_heads[..panel_cols]);
    assert_eq!(betas, expected_betas[..panel_cols]);
    assert_eq!(tail_heads, expected_heads[panel_cols..]);
    assert_eq!(tail_betas, expected_betas[panel_cols..]);
}

#[test]
fn packed_qr_panel_application_validates_shapes_and_values() {
    let mut trailing = [1.0f32, 2.0];
    let error = apply_packed_qr_panel_left(
        &[1.0, 0.0, 0.0, 1.0],
        2,
        2,
        &[1.0],
        &[1.0, 1.0],
        &mut trailing,
        1,
    )
    .expect_err("short heads must be rejected");
    assert!(matches!(
        error,
        HephaestusError::InvalidConfiguration { message }
            if message.contains("heads length 1")
    ));

    trailing[1] = f32::INFINITY;
    let error = apply_packed_qr_panel_left(
        &[1.0, 0.0, 0.0, 1.0],
        2,
        2,
        &[1.0, 1.0],
        &[1.0, 1.0],
        &mut trailing,
        1,
    )
    .expect_err("non-finite trailing input must be rejected");
    assert!(matches!(
        error,
        HephaestusError::DispatchFailed { message }
            if message.contains("non-finite")
    ));
}

// ── split_packed_lu ──────────────────────────────────────────────

#[test]
fn split_packed_lu_extracts_unit_lower_and_upper() {
    // Packed 3×3 factor storage holding
    //   L (strict lower, implicit unit diagonal): [[·,·,·],[4,·,·],[7,8,·]]
    //   U (upper, including diagonal):            [[1,2,3],[·,5,6],[·,·,9]]
    #[rustfmt::skip]
    let packed = vec![
        1.0f32, 2.0, 3.0,
        4.0,    5.0, 6.0,
        7.0,    8.0, 9.0,
    ];
    let (l, u) = split_packed_lu(&packed, 3).expect("invariant: packed.len() == 9 matches n = 3");
    // The split is a pure copy of dyadic values: exact equality is the
    // correct oracle (no arithmetic occurs).
    #[rustfmt::skip]
    assert_eq!(l, vec![
        1.0, 0.0, 0.0,
        4.0, 1.0, 0.0,
        7.0, 8.0, 1.0,
    ]);
    #[rustfmt::skip]
    assert_eq!(u, vec![
        1.0, 2.0, 3.0,
        0.0, 5.0, 6.0,
        0.0, 0.0, 9.0,
    ]);
}

#[test]
fn split_packed_lu_composes_with_panel_lu() {
    // A = [[2, 1], [4, 3]]: factorise, split, then verify L·U = P·A.
    // All entries and intermediates (factor 2/4 = 0.5, products, sums)
    // are dyadic rationals, exact in binary floating point, so exact
    // equality is the analytically correct oracle.
    let original = vec![2.0f32, 1.0, 4.0, 3.0];
    let mut packed = original.clone();
    let pivots = panel_lu_packed(&mut packed, 2).expect("invariant: nonsingular 2x2 factorises");
    let (l, u) = split_packed_lu(&packed, 2).expect("invariant: packed.len() == 4 matches n = 2");

    // Apply the recorded transpositions to obtain P·A.
    let mut pa = original;
    for (k, &p) in pivots.iter().enumerate() {
        if p != k {
            for j in 0..2 {
                pa.swap(k * 2 + j, p * 2 + j);
            }
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            let mut s = 0.0f32;
            for k in 0..2 {
                s += l[i * 2 + k] * u[k * 2 + j];
            }
            assert_eq!(
                s,
                pa[i * 2 + j],
                "L·U[{i},{j}] must reproduce P·A exactly for dyadic input"
            );
        }
    }
}

#[test]
fn split_packed_lu_length_mismatch_is_rejected() {
    let packed = vec![1.0f32, 2.0, 3.0]; // 3 elements, but n = 2 requires 4
    assert!(matches!(
        split_packed_lu(&packed, 2),
        Err(HephaestusError::LengthMismatch {
            host_len: 4,
            device_len: 3
        })
    ));
}
