use super::super::{apply_packed_qr_panel_left, panel_qr_packed};
use crate::domain::error::HephaestusError;

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
