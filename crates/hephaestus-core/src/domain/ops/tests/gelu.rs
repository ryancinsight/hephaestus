use super::*;

/// `GeluTanhGradOp` overflows on a *finite* input once `three * c1 * x * x`
/// itself exceeds the format's range (`|x| ≈ 5.04e19` in `f32`, `≈ 3.7e154`
/// in `f64`). By that magnitude the inner sigmoid has already saturated to
/// exactly `0`/`1` bit-exactly (its argument overflows to `±∞` through the
/// cubic term at a much smaller `|x|` — `c1 * x³` alone exceeds `f32::MAX`
/// past `|x| ≈ 1.97e13`), so the vanished `s(1-s)` factor multiplies an
/// overflowing `x²` (`0 · ∞ = NaN`) rather than the guard below returning
/// the already-saturated `s` directly. Fails against ca3c28c, which returns
/// `NaN` at every input here (confirmed before applying the fix).
#[test]
fn gelu_tanh_grad_resolves_the_overflowing_finite_tail() {
    assert_eq!(GeluTanhGradOp::apply(5.04e19_f32), 1.0);
    assert_eq!(GeluTanhGradOp::apply(-5.04e19_f32), 0.0);
    assert_eq!(GeluTanhGradOp::apply(3.7e154_f64), 1.0);
    assert_eq!(GeluTanhGradOp::apply(-3.7e154_f64), 0.0);
}

/// `GeluTanhGradOp` switches from `1 - s` to `sigmoid(-w)` above
/// `w = 2z = 2.47`. The cases below the switch (`w ≈ 2.31`/`2.35`) kill an
/// always-independent form and any threshold at or below them; the cases
/// above it (`w ≈ 2.68`/`2.70`) kill an always-direct form and any threshold
/// at or above them, so a threshold moved by 10% either way fails; the tail cases are the original direct-form
/// cancellation, where `s` has rounded to `1`.
#[test]
fn gelu_tanh_grad_selects_the_more_accurate_form_on_each_side_of_its_crossover() {
    assert_accuracy_cases(
        "GeluTanhGrad",
        GeluTanhGradOp::apply,
        f32::MANTISSA_DIGITS,
        &[
            // 0.148 / 1.852 ULP
            AccuracyCase {
                input: 1.340_434_f32,
                reference: (1.127_676_469_325_747_5, 1.007_740_701_404_698_9e-16),
                bound_ulp: 1.0,
            },
            // 0.089 / 1.911 ULP
            AccuracyCase {
                input: 1.520_734_1_f32,
                reference: (1.127_006_281_717_678, -1.096_801_909_540_929_4e-18),
                bound_ulp: 1.0,
            },
            // 0.743 / 16.257 ULP
            AccuracyCase {
                input: 4.96_f32,
                reference: (1.000_001_995_905_308_4, -7.896_238_384_062_334e-17),
                bound_ulp: 2.0,
            },
        ],
    );
    assert_accuracy_cases(
        "GeluTanhGrad",
        GeluTanhGradOp::apply,
        f64::MANTISSA_DIGITS,
        &[
            // 0.172 / 1.828 ULP
            AccuracyCase {
                input: 1.359_537_479_066_083_7_f64,
                reference: (1.128_251_526_320_643_6, 3.824_062_359_236_039_5e-17),
                bound_ulp: 1.0,
            },
            // 0.015 / 2.015 ULP
            AccuracyCase {
                input: 1.530_396_691_565_789_6_f64,
                reference: (1.126_630_199_140_305_6, -3.437_292_819_137_683e-18),
                bound_ulp: 1.0,
            },
            // 0.047 / 42.953 ULP
            AccuracyCase {
                input: 7.09_f64,
                reference: (1.000_000_000_000_009_5, -1.037_975_140_742_717_5e-17),
                bound_ulp: 1.0,
            },
        ],
    );
}

/// Within `±0.05` of its root the two terms of `GeluTanhGrad` cancel, so an
/// error is measured in ULP of the dominant term `s = σ(2z)` (the result's
/// own ULP shrinks without bound at the root). Every 256th `f32` input of the
/// window and 4,096 evenly spaced `f64` inputs are checked against the
/// double-double reference.
///
/// Bound basis: the largest measured errors of the implementation over the
/// whole window are 4.681 dominant-term ULP over every `f32` input and 4.293
/// over 10,000,000 uniform `f64` samples (five seeds); `5.0` admits both with
/// margin and rejects an error of more than a third of an ULP beyond them.
#[test]
fn gelu_tanh_grad_stays_within_the_measured_bound_near_its_root() {
    const BOUND_DOMINANT_ULP: f64 = 5.0;
    let low = GELU_TANH_GRAD_ROOT - 0.05;
    let high = GELU_TANH_GRAD_ROOT + 0.05;
    // Negative `f32` bit patterns grow with magnitude.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the window bounds are representable to within one f32 rounding"
    )]
    let (first, last) = ((high as f32).to_bits(), (low as f32).to_bits());
    let mut worst32 = 0.0_f64;
    for bits in (first..=last).step_by(256) {
        let x = f32::from_bits(bits);
        let got = f64::from(GeluTanhGradOp::apply(x));
        let error = gelu_tanh_grad_dominant_term_ulps(f64::from(x), got, f32::MANTISSA_DIGITS);
        worst32 = worst32.max(error);
    }
    assert!(
        worst32 <= BOUND_DOMINANT_ULP,
        "f32 GeluTanhGrad near its root: {worst32} dominant-term ULP"
    );
    let samples = 4096_u32;
    let mut worst64 = 0.0_f64;
    for index in 0..samples {
        let x = low + (high - low) * (f64::from(index) + 0.5) / f64::from(samples);
        let got = GeluTanhGradOp::apply(x);
        let error = gelu_tanh_grad_dominant_term_ulps(x, got, f64::MANTISSA_DIGITS);
        worst64 = worst64.max(error);
    }
    assert!(
        worst64 <= BOUND_DOMINANT_ULP,
        "f64 GeluTanhGrad near its root: {worst64} dominant-term ULP"
    );
}

/// Each `GeluTanhGradOp` branch keeps the multiplication order of the
/// single-form evaluation it is measured against, and the other order raises
/// the `f32` window maximum (see the operator's documentation). Over every
/// `f32` input of each window below, the shipped order's maximum and the
/// other order's were measured against a double-double reference; each
/// bound lies between the two.
///
/// - `[0.515, 0.555]`, direct branch: 1.784 ULP factor by factor (shipped),
///   1.936 pre-grouped. Bound 1.86.
/// - `[1.625, 1.725]`, independent branch: 1.681 ULP pre-grouped (shipped),
///   1.774 factor by factor. Bound 1.73.
#[test]
fn gelu_tanh_grad_multiplication_orders_hold_their_measured_window_maxima() {
    let direct = gelu_tanh_grad_f32_window_maximum(0.515, 0.555);
    assert!(direct <= 1.86, "direct-branch window maximum {direct} ULP");
    let independent = gelu_tanh_grad_f32_window_maximum(1.625, 1.725);
    assert!(
        independent <= 1.73,
        "independent-branch window maximum {independent} ULP"
    );
}
