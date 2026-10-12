use super::*;

/// `MishGradOp` switches from `1 - t²` to `sech²(sp)` above `sp = 1.55`. The
/// cases below the switch (`sp ≈ 1.03`/`1.11`, and `1.400`/`1.54999` just
/// inside it) kill an always-`sech²` form and any threshold below them,
/// including `1.55` moved down by 10% (`1.395`); the cases above it
/// (`sp ≈ 1.5503`/`1.605`, and `1.96`/`1.97`) kill an always-direct form
/// and any threshold at or above them, including `1.55` moved up by 10%
/// (`1.705`). The two forms differ by one ULP just inside the switch, so
/// those cases require the selected form to be correctly rounded (half an
/// ULP). The tail cases are the original direct-form cancellation, where
/// `t` has rounded to `1`.
#[test]
fn mish_grad_selects_the_more_accurate_form_on_each_side_of_its_crossover() {
    assert_accuracy_cases(
        "MishGrad",
        MishGradOp::apply,
        f32::MANTISSA_DIGITS,
        &[
            // 0.003 / 2.003 ULP
            AccuracyCase {
                input: 0.589_513_8_f32,
                reference: (9.261_161_090_871_325e-1, 2.060_666_313_517_767_4e-17),
                bound_ulp: 1.0,
            },
            // 0.009 / 1.009 ULP
            AccuracyCase {
                input: 1.116_696_7_f32,
                reference: (1.067_210_794_586_083_1, 6.519_837_544_120_851e-17),
                bound_ulp: 0.5,
            },
            // 0.015 / 1.015 ULP
            AccuracyCase {
                input: 1.311_765_7_f32,
                reference: (1.084_256_766_469_009_3, 2.719_817_198_156_410_7e-18),
                bound_ulp: 0.5,
            },
            // 0.447 / 1.553 ULP
            AccuracyCase {
                input: 1.810_998_3_f32,
                reference: (1.079_494_768_014_944_7, 6.340_200_556_247_342e-17),
                bound_ulp: 1.0,
            },
            // 0.261 / 4.739 ULP
            AccuracyCase {
                input: 9.01_f32,
                reference: (1.000_000_507_972_837_8, -1.054_899_121_256_476_9e-17),
                bound_ulp: 1.0,
            },
        ],
    );
    assert_accuracy_cases(
        "MishGrad",
        MishGradOp::apply,
        f64::MANTISSA_DIGITS,
        &[
            // 0.016 / 2.016 ULP
            AccuracyCase {
                input: 0.707_785_371_182_024_1_f64,
                reference: (9.715_328_789_566_628e-1, -1.785_604_082_853_674_5e-18),
                bound_ulp: 1.0,
            },
            // 0.015 / 1.015 ULP
            AccuracyCase {
                input: 1.311_411_733_274_786_6_f64,
                reference: (1.084_238_829_570_115, 3.254_164_456_260_444_2e-18),
                bound_ulp: 0.5,
            },
            // 0.010 / 1.010 ULP
            AccuracyCase {
                input: 1.380_909_773_517_28_f64,
                reference: (1.086_984_743_270_022_8, -2.295_144_350_268_792_2e-18),
                bound_ulp: 0.5,
            },
            // 0.430 / 1.570 ULP
            AccuracyCase {
                input: 1.814_309_182_963_652_4_f64,
                reference: (1.079_334_820_511_441_7, -9.549_391_746_980_466e-17),
                bound_ulp: 1.0,
            },
            // 0.678 / 9.322 ULP
            AccuracyCase {
                input: 19.04_f64,
                reference: (1.000_000_000_000_002_2, -7.144_888_117_181_882e-17),
                bound_ulp: 2.0,
            },
        ],
    );
}
