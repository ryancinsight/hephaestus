use super::*;

/// `SiluGradOp` switches from `1 - sigmoid(x)` to `sigmoid(-x)` above
/// `x = 2.63`. The cases below the switch kill an always-independent form and
/// any threshold at or below `2.40` (`f32`) / `2.55` (`f64`); the cases above
/// it kill an always-direct form and any threshold at or above `2.87` /
/// `2.90`, so a threshold moved by 10% either way fails; the tail cases are the original direct-form cancellation, where
/// `sigmoid(x)` has rounded so close to `1` that `1 - sigmoid(x)` keeps few
/// significant bits.
#[test]
fn silu_grad_selects_the_more_accurate_form_on_each_side_of_its_crossover() {
    assert_accuracy_cases(
        "SiluGrad",
        SiluGradOp::apply,
        f32::MANTISSA_DIGITS,
        &[
            // 0.115 / 1.885 ULP
            AccuracyCase {
                input: 2.405_141_8_f32,
                reference: (1.099_837_793_692_915_3, 6.976_311_581_976_778e-17),
                bound_ulp: 1.0,
            },
            // 0.017 / 2.017 ULP
            AccuracyCase {
                input: 2.867_271_7_f32,
                reference: (1.092_152_835_918_219_5, -6.123_557_629_831_417e-17),
                bound_ulp: 1.0,
            },
            // 0.215 / 7.785 ULP
            AccuracyCase {
                input: 16.64_f32,
                reference: (1.000_000_928_061_517_4, 1.188_321_631_270_538_1e-17),
                bound_ulp: 1.0,
            },
        ],
    );
    assert_accuracy_cases(
        "SiluGrad",
        SiluGradOp::apply,
        f64::MANTISSA_DIGITS,
        &[
            // 0.091 / 1.909 ULP
            AccuracyCase {
                input: 2.552_028_415_350_948_f64,
                reference: (1.098_860_025_164_296_6, -2.016_099_943_032_254_3e-17),
                bound_ulp: 1.0,
            },
            // 0.009 / 2.009 ULP
            AccuracyCase {
                input: 2.903_298_488_057_376_7_f64,
                reference: (1.091_106_185_802_179_8, 1.901_338_674_739_636_7e-18),
                bound_ulp: 1.0,
            },
            // 0.987 / 18.013 ULP
            AccuracyCase {
                input: 36.73_f64,
                reference: (1.000_000_000_000_004, -2.907_402_465_278_531_4e-18),
                bound_ulp: 2.0,
            },
        ],
    );
}
