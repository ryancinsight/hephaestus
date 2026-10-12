use super::super::{require_dense_operand, validate_square_operand};
use crate::domain::error::HephaestusError;
use leto::Layout;

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
