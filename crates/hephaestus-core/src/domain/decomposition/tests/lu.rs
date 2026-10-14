use super::*;

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
