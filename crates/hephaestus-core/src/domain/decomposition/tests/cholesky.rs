use super::*;

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
