//! Tail-accuracy regressions for the stable formulations above.

use super::super::{GeluOp, MishOp, UnaryValue};
use super::stable::{softplus_value, stable_sigmoid};

/// Derivation: `f64`'s `exp`/`ln_1p` are accurate to a few ULP, so a
/// `1e-6` *relative* tolerance bounds every assertion below far above
/// that rounding noise while catching a cancellation collapse to `0`
/// outright (a collapse is 100% relative error). An *absolute* `1e-6`
/// tolerance is unsound here: `softplus(-20) ≈ 2.06e-9` and
/// `mish(-20) ≈ -4.12e-8` are themselves smaller than `1e-6`, so an
/// absolute bound passes exactly the collapse-to-zero defect these two
/// tests exist to catch (the test integrity gap ADR 0061 Decision 6
/// exists to close).
const REL_TOL: f64 = 1e-6;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= REL_TOL * b.abs()
}

#[test]
fn softplus_matches_the_cancellation_and_overflow_cases() {
    // softplus(-20) = ln(1 + exp(-20)); exp(-20) = 2.061e-9 is far below
    // f64 ULP(1), so ln(1+e) ~ e to full relative precision, giving the
    // reference 2.061153622e-9 (ADR 0061 Decision 6). A collapse to
    // exactly 0 (the defect this test guards) is 100% relative error,
    // caught by the relative bound above regardless of the reference's
    // own tiny magnitude.
    assert!(close(softplus_value(-20.0f64), 2.061_153_622e-9,));
    // softplus(100) = 100 + ln_1p(exp(-100)); exp(-100) underflows f64's
    // ability to move 100.0 at all (ULP(100) ~ 1.4e-14 >> exp(-100)).
    assert_eq!(softplus_value(100.0f64), 100.0);
}

/// Mutation-killing regression: a mutant substituting the naive
/// `ln(1 + exp(x))` for the `ln_1p`-based form collapses to exactly `0`
/// here. `exp(-100) ≈ 3.72e-44` is far below `f64`'s `ulp(1) ≈
/// 1.11e-16`, so `1.0 + exp(-100)` rounds to exactly `1.0` and
/// `ln(1.0) == 0.0` — 100% relative error, which the module's own
/// `REL_TOL = 1e-6` test above (at `x = -20`, where `exp(-20) ≈
/// 2.06e-9` is still far above `f64`'s ULP) would already catch, but
/// this test exercises the collapse at the much more extreme magnitude
/// the mutation search actually found. `softplus(x) = x + ln(1+e^x)`
/// for the negated branch gives `softplus(-100) = ln(1+e^-100) =
/// e^-100 - e^-200/2 + O(e^-300)`; the dropped `O(e^-200)` term is
/// about 40 orders of magnitude below `1e-12` relative to the leading
/// `e^-100` term, so `softplus(-100)` must match `exp(-100)` to far
/// tighter than `1e-12` relative — a bound the naive mutant's exact-`0`
/// output fails outright.
#[test]
fn softplus_negative_100_matches_exp_within_1e_minus_12_relative() {
    let got = softplus_value(-100.0f64);
    let reference = (-100.0f64).exp();
    let relative_error = (got - reference).abs() / reference.abs();
    assert!(
        relative_error < 1e-12,
        "softplus(-100) = {got:e}, exp(-100) = {reference:e}, relative error {relative_error:e}"
    );
}

#[test]
fn stable_sigmoid_avoids_the_overflow_collapse() {
    // sigmoid(-89) = 1/(1+exp(89)); exp(89) overflows f32 (> 3.4e38), so
    // the naive quotient rounds to 0 exactly. The true value, exp(-89) to
    // leading order, is a normal f32 (~1.97e-39 is representable; the
    // silu product below lands in the exponent range that matters).
    let s = stable_sigmoid(-89.0f32);
    assert!(s > 0.0, "sigmoid(-89) must be a nonzero normal float");
    let silu = -89.0f32 * s;
    assert!(silu != 0.0, "silu(-89) must not underflow to zero");
}

#[test]
fn mish_reproduces_the_negative_tail_reference() {
    // mish(-20) = -20 * tanh(softplus(-20)); tanh is linear near 0, so
    // tanh(2.061e-9) ~ 2.061e-9, giving -4.122e-8 (ADR 0061 Decision 6:
    // "Mish(-20) 0 against -4.12e-8" names this as the rendering defect
    // the value function must not repeat).
    let m = MishOp::apply(-20.0f64);
    assert!(close(m, -4.122_307_244e-8));
}

#[test]
fn gelu_matches_the_erfc_identity_in_the_negative_tail() {
    // gelu(-10) = 0.5 * -10 * erfc(10/sqrt(2)); erfc(7.07) is a tiny but
    // normal double (~7.6e-13), so gelu(-10) must be a nonzero normal
    // float, never the zero a naive 1+erf(-z) cancellation would give.
    let g = GeluOp::apply(-10.0f64);
    assert!(g != 0.0, "gelu(-10) must be a nonzero normal float");
    assert!(g < 0.0, "gelu(-10) must keep the sign of a negative input");
}

#[test]
fn round_trips_the_gelu_reference_at_zero_and_large_positive() {
    assert_eq!(GeluOp::apply(0.0f64), 0.0);
    // gelu(x) -> x as x -> +inf: erfc(-large) -> 2, so 0.5*x*2 = x.
    assert!(close(GeluOp::apply(10.0f64), 10.0));
}
