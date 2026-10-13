use super::{accuracy_bound, relative_error};
use crate::application::canary::{CanaryWidth, MathBuiltin, PrecisionGrade, PrecisionReport};

const MODELS: [(MathBuiltin, u128); 4] = [
    (MathBuiltin::Exp, 43),
    (MathBuiltin::Log, 9),
    (MathBuiltin::Pow, 99),
    (MathBuiltin::Tanh, 252),
];

/// Compare the measured binary64 field with an exact rational using integer
/// arithmetic. These ratios need at most 107 bits, so u128 cannot overflow.
fn encloses_ratio(value: f64, numerator: u128, denominator: u128) -> bool {
    let bits = value.to_bits();
    let exponent = i32::try_from((bits >> 52) & 0x7ff).expect("eleven exponent bits") - 1023 - 52;
    assert!(exponent < 0);
    let shift = u32::try_from(-exponent).expect("negative exponent magnitude");
    let significand = u128::from((bits & ((1_u64 << 52) - 1)) | (1_u64 << 52));
    significand
        .checked_mul(denominator)
        .expect("107-bit comparison")
        >= numerator
            .checked_mul(1_u128.checked_shl(shift).expect("ratio exponent fits"))
            .expect("107-bit comparison")
}

#[test]
fn calibration_bounds_enclose_exact_ratios_at_both_widths() {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(
            tracing_subscriber::fmt::format::FmtSpan::ENTER
                | tracing_subscriber::fmt::format::FmtSpan::CLOSE,
        )
        .finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let _arrange = tracing::info_span!("arrange").entered();
    for (builtin, budget) in MODELS {
        let act = tracing::info_span!("act", ?builtin).entered();
        let narrow = accuracy_bound::<f32>(builtin);
        let wide = accuracy_bound::<f64>(builtin);
        drop(act);
        let _assert = tracing::info_span!("assert", ?builtin).entered();
        let denominator = (1_u128 << 52) - budget;
        // eps32 / eps64 = 2^29; the denominator scales by 2^52.
        for (bound, numerator) in [(narrow, budget * ((1_u128 << 29) + 1)), (wide, budget * 2)] {
            assert!(encloses_ratio(bound, numerator, denominator));
            // RN division followed by next_up is at most two adjacent steps
            // above the exact rational, independent of the observed GPU error.
            assert!(!encloses_ratio(
                bound.next_down().next_down(),
                numerator,
                denominator
            ));
        }
    }
}

#[test]
fn calibration_grades_reject_errors_outside_their_proven_bands() {
    let subscriber = tracing_subscriber::fmt().with_test_writer().finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    for (builtin, _) in MODELS {
        let arrange = tracing::info_span!("arrange", ?builtin).entered();
        let narrow = accuracy_bound::<f32>(builtin);
        let wide = accuracy_bound::<f64>(builtin);
        let report = |width, error| PrecisionReport {
            builtin,
            width,
            samples: 8,
            max_relative_error: error,
        };
        drop(arrange);
        let act = tracing::info_span!("act", ?builtin).entered();
        let boundary_grades = [
            report(CanaryWidth::F32, narrow.next_down()).grade(),
            report(CanaryWidth::F32, narrow).grade(),
            report(CanaryWidth::F32, narrow.next_up()).grade(),
            report(CanaryWidth::F64, wide).grade(),
            report(CanaryWidth::F64, wide.next_up()).grade(),
            report(CanaryWidth::F64, narrow).grade(),
            report(CanaryWidth::F64, narrow.next_up()).grade(),
        ];
        drop(act);
        let _assert = tracing::info_span!("assert", ?builtin).entered();
        assert_eq!(
            boundary_grades,
            [
                PrecisionGrade::Exact,
                PrecisionGrade::Exact,
                PrecisionGrade::Poor,
                PrecisionGrade::Exact,
                PrecisionGrade::SingleGrade,
                PrecisionGrade::SingleGrade,
                PrecisionGrade::Poor,
            ]
        );
        for error in [-f64::EPSILON, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for width in [CanaryWidth::F32, CanaryWidth::F64] {
                assert_eq!(report(width, error).grade(), PrecisionGrade::Poor);
            }
        }
    }
}

#[test]
fn relative_measurement_preserves_dyadic_values_and_exact_zero_identities() {
    let subscriber = tracing_subscriber::fmt().with_test_writer().finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let arrange = tracing::info_span!("arrange").entered();
    let finite = [
        (2.0_f64, 2.0, 0.0_f64),
        (3.0, 2.0, 0.5),
        (-1.0, -2.0, 0.5),
        (0.0, -0.0, 0.0),
    ];
    let invalid = [
        (f64::from_bits(1), 0.0),
        (-f64::from_bits(1), 0.0),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (f64::INFINITY, 1.0),
        (1.0, f64::INFINITY),
    ];
    drop(arrange);
    for (actual, reference, expected) in finite {
        let act = tracing::info_span!("act", actual, reference).entered();
        let error = relative_error(actual, reference);
        drop(act);
        let _assert = tracing::info_span!("assert").entered();
        assert_eq!(error.to_bits(), expected.to_bits());
    }
    for (actual, reference) in invalid {
        let act = tracing::info_span!("act", actual, reference).entered();
        let error = relative_error(actual, reference);
        drop(act);
        let _assert = tracing::info_span!("assert").entered();
        assert_eq!(error, f64::INFINITY);
    }
}
