use super::*;

/// `TanhGradOp` switches from `1 - y*y` to `(1 - y) * (1 + y)` above
/// `|y| = 0.75`. The small-`|y|` cases kill an always-factored form; the
/// cases just above `√½` (`0.7071`) kill any threshold at or below them,
/// including the previous `½` and `0.75` moved down by 10% (`0.675`); the
/// cases at `0.778`/`0.8125` kill any threshold at or above them, including
/// `0.75` moved up by 10% (`0.825`), and those near `0.875`/`0.9` an
/// always-direct form; the cases near
/// `1` are the original direct-form cancellation. Where the two forms differ
/// by exactly one ULP the bound is half an ULP: the selected form must be
/// correctly rounded.
#[test]
fn tanh_grad_selects_the_more_accurate_form_on_each_side_of_its_crossover() {
    assert_accuracy_cases(
        "TanhGrad",
        TanhGradOp::apply,
        f32::MANTISSA_DIGITS,
        &[
            // 0.500 / 1.500 ULP
            AccuracyCase {
                input: 0.003_625_303_7_f32,
                reference: (9.999_868_571_727_95e-1, 4.157_915_331_481_909_5e-17),
                bound_ulp: 1.0,
            },
            // 0.000 / 1.000 ULP
            AccuracyCase {
                input: 0.500_000_06_f32,
                reference: (7.499_999_403_953_517e-1, 0.0),
                bound_ulp: 0.5,
            },
            // 0.015 / 1.015 ULP
            AccuracyCase {
                input: 0.707_112_13_f32,
                reference: (4.999_924_306_528_918e-1, 0.0),
                bound_ulp: 0.5,
            },
            // 0.000 / 1.000 ULP
            AccuracyCase {
                input: 0.812_500_24_f32,
                reference: (3.398_433_625_697_521e-1, 0.0),
                bound_ulp: 0.5,
            },
            // 0.000 / 2.000 ULP
            AccuracyCase {
                input: 0.874_999_9_f32,
                reference: (2.343_752_086_162_425e-1, 0.0),
                bound_ulp: 1.0,
            },
            // 0.299 / 989.299 ULP
            AccuracyCase {
                input: 0.99988_f32,
                reference: (2.399_539_036_694_875_4e-4, 0.0),
                bound_ulp: 1.0,
            },
        ],
    );
    assert_accuracy_cases(
        "TanhGrad",
        TanhGradOp::apply,
        f64::MANTISSA_DIGITS,
        &[
            // 0.433 / 1.433 ULP
            AccuracyCase {
                input: 0.005_575_415_809_682_238_f64,
                reference: (9.999_689_147_385_491e-1, 4.808_746_349_255_426_6e-17),
                bound_ulp: 1.0,
            },
            // 0.013 / 1.013 ULP
            AccuracyCase {
                input: 0.733_984_678_348_328_f64,
                reference: (4.612_664_919_499_014_4e-1, -7.084_623_958_970_427e-19),
                bound_ulp: 0.5,
            },
            // 0.000 / 1.000 ULP
            AccuracyCase {
                input: 0.777_782_521_737_601_f64,
                reference: (3.950_543_488_794_982_5e-1, 8.953_235_885_114_249e-25),
                bound_ulp: 0.5,
            },
            // 0.000 / 2.000 ULP
            AccuracyCase {
                input: 0.899_806_573_745_678_2_f64,
                reference: (1.903_481_298_440_634e-1, 2.693_694_929_135_220_4e-23),
                bound_ulp: 1.0,
            },
            // 0.015 / 27134340.015 ULP
            AccuracyCase {
                input: 0.999_999_993_3_f64,
                reference: (1.339_999_995_360_794_7e-8, -2.509_200_139_160_843_5e-26),
                bound_ulp: 1.0,
            },
        ],
    );
}
