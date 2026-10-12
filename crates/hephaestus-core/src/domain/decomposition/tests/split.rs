use super::super::{panel_lu_packed, split_packed_lu};
use crate::domain::error::HephaestusError;

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
