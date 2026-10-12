use super::*;

/// `HardswishOp` computed `x * clamp / 6`: for `x ≥ 3` the clamp saturates
/// to `6` and the true value is `x` exactly, but `x * 6` overflows before
/// the `/ 6` brings it back down once `x` exceeds `f32::MAX / 6 ≈ 5.67e37`
/// (`f64::MAX / 6 ≈ 3.0e307`). Dividing the clamp by `6` before multiplying
/// keeps every intermediate value at most `1`. Fails against ca3c28c, which
/// returns `inf` at both inputs (confirmed before applying the fix).
#[test]
fn hardswish_does_not_overflow_once_the_clamp_saturates() {
    assert_eq!(HardswishOp::apply(1e38_f32), 1e38_f32);
    assert_eq!(HardswishOp::apply(1e308_f64), 1e308_f64);
}

/// `HardswishOp` at `x = -∞`: the clamp saturates to `0` there too, but the
/// literal `x * (0 / 6)` forms `-∞ · 0 = NaN` — a second-pass defect the
/// first fix (6ea4acd) left in place, since its `-∞`/`+∞` audit covered only
/// the sigmoid/tanh-based operators. Fails against 6ea4acd (and ca3c28c),
/// both of which return `NaN`.
#[test]
fn hardswish_resolves_negative_infinity_to_negative_zero() {
    let got32 = HardswishOp::apply(f32::NEG_INFINITY);
    assert!(
        got32 == 0.0 && got32.is_sign_negative(),
        "Hardswish(-inf) must be -0 in f32, got {got32:?}"
    );
    let got64 = HardswishOp::apply(f64::NEG_INFINITY);
    assert!(
        got64 == 0.0 && got64.is_sign_negative(),
        "Hardswish(-inf) must be -0 in f64, got {got64:?}"
    );
}
