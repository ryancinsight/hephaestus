use super::*;

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
