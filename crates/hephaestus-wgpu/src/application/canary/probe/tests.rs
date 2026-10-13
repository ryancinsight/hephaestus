use core::fmt::Debug;

use super::{MathBuiltin, PROBE_LEN, ScalarProbe, probe_inputs, reference_value};

type InputContract<T> = [(MathBuiltin, [T; PROBE_LEN], [T; PROBE_LEN]); 4];

fn verify_inputs<T: ScalarProbe, B: Debug + PartialEq>(
    expected: InputContract<T>,
    bits: impl Fn(T) -> B,
) {
    for (builtin, lhs, rhs) in expected {
        let act = tracing::info_span!("act", ?builtin).entered();
        let actual = probe_inputs::<T>(builtin);
        drop(act);
        let _assert = tracing::info_span!("assert", ?builtin).entered();
        assert_eq!(actual.0.map(&bits), lhs.map(&bits));
        assert_eq!(actual.1.map(&bits), rhs.map(&bits));
    }
}

#[test]
fn native_inputs_preserve_original_bit_patterns() {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(
            tracing_subscriber::fmt::format::FmtSpan::ENTER
                | tracing_subscriber::fmt::format::FmtSpan::CLOSE,
        )
        .finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let arrange = tracing::info_span!("arrange").entered();
    // Published IEEE constants and the original eight samples per builtin.
    let narrow: InputContract<f32> = [
        (
            MathBuiltin::Exp,
            [1.0, 0.5, -1.0, 2.0, 10.0, -10.0, 0.0, 20.0],
            [0.0; PROBE_LEN],
        ),
        (
            MathBuiltin::Log,
            [
                1.0,
                core::f32::consts::E,
                0.5,
                2.0,
                10.0,
                100.0,
                0.1,
                1000.0,
            ],
            [0.0; PROBE_LEN],
        ),
        (
            MathBuiltin::Pow,
            [2.0, 9.0, core::f32::consts::E, 4.0, 1.5, 10.0, 0.5, 3.0],
            [10.0, 0.5, 1.0, 3.0, 2.5, 2.0, 0.5, 3.0],
        ),
        (
            MathBuiltin::Tanh,
            [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 10.0],
            [0.0; PROBE_LEN],
        ),
    ];
    let wide: InputContract<f64> = [
        (
            MathBuiltin::Exp,
            [1.0, 0.5, -1.0, 2.0, 10.0, -10.0, 0.0, 20.0],
            [0.0; PROBE_LEN],
        ),
        (
            MathBuiltin::Log,
            [
                1.0,
                core::f64::consts::E,
                0.5,
                2.0,
                10.0,
                100.0,
                0.1,
                1000.0,
            ],
            [0.0; PROBE_LEN],
        ),
        (
            MathBuiltin::Pow,
            [2.0, 9.0, core::f64::consts::E, 4.0, 1.5, 10.0, 0.5, 3.0],
            [10.0, 0.5, 1.0, 3.0, 2.5, 2.0, 0.5, 3.0],
        ),
        (
            MathBuiltin::Tanh,
            [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 10.0],
            [0.0; PROBE_LEN],
        ),
    ];
    drop(arrange);
    verify_inputs(narrow, f32::to_bits);
    verify_inputs(wide, f64::to_bits);
}

#[test]
fn reference_values_retain_native_libm_bits() {
    let subscriber = tracing_subscriber::fmt().with_test_writer().finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let arrange = tracing::info_span!("arrange").entered();
    let cases = [
        (MathBuiltin::Exp, -1.0_f64, 0.0, (-1.0_f64).exp()),
        (MathBuiltin::Log, 0.5, 0.0, 0.5_f64.ln()),
        (MathBuiltin::Pow, 9.0, 0.5, 9.0_f64.powf(0.5)),
        (MathBuiltin::Tanh, -0.5, 0.0, (-0.5_f64).tanh()),
    ];
    drop(arrange);
    for (builtin, lhs, rhs, expected) in cases {
        let act = tracing::info_span!("act", ?builtin).entered();
        let got = reference_value(builtin, lhs, rhs);
        drop(act);
        let _assert = tracing::info_span!("assert", ?builtin).entered();
        assert_eq!(got.to_bits(), expected.to_bits());
    }
}
