use super::*;

#[test]
fn width_of_covers_floats_only() {
    assert_eq!(width_of::<f32>(), Some(CanaryWidth::F32));
    assert_eq!(width_of::<f64>(), Some(CanaryWidth::F64));
    assert_eq!(width_of::<u32>(), None);
    assert_eq!(width_of::<i32>(), None);
}

#[test]
fn scan_finds_builtin_calls_in_templates() {
    assert_eq!(scan_builtins("exp(x)"), vec![MathBuiltin::Exp]);
    assert_eq!(
        scan_builtins("log(x) * 0.43429448190325182"),
        vec![MathBuiltin::Log]
    );
    assert_eq!(scan_builtins("pow(a, b)"), vec![MathBuiltin::Pow]);
    assert_eq!(scan_builtins("tanh(x)"), vec![MathBuiltin::Tanh]);
    // Composite expressions carry their builtins with them.
    assert_eq!(
        scan_builtins(
            "0.5 * (x) * (1.0 + tanh(0.7978845608028654 * ((x) + 0.044715 * (x) * (x) * (x))))"
        ),
        vec![MathBuiltin::Tanh]
    );
    assert_eq!(scan_builtins("(exp(x) - 1.0)"), vec![MathBuiltin::Exp]);
}

#[test]
fn scan_rejects_lookalike_prefixes() {
    // `atanh(` contains `tanh(`; the boundary check must reject it.
    assert!(scan_builtins("atanh(x)").is_empty());
    // Suffix builtins share the `log` stem but are distinct calls.
    assert!(scan_builtins("log2(x)").is_empty());
    assert!(scan_builtins("log10(x)").is_empty());
    assert!(scan_builtins("lgamma(x)").is_empty());
    assert!(scan_builtins("exp2(x)").is_empty());
    // Plain arithmetic carries no builtins.
    assert!(scan_builtins("(a) + (b)").is_empty());
    assert!(scan_builtins("select(0.0, 1.0, (x) > 0.0)").is_empty());
}

#[test]
fn grades_cut_between_exact_single_and_poor() {
    let report = |width, err| PrecisionReport {
        builtin: MathBuiltin::Exp,
        width,
        samples: 8,
        max_relative_error: err,
    };
    assert_eq!(
        report(CanaryWidth::F64, 1e-15).grade(),
        PrecisionGrade::Exact
    );
    assert_eq!(
        report(CanaryWidth::F64, 5.96e-8).grade(),
        PrecisionGrade::SingleGrade
    );
    assert_eq!(report(CanaryWidth::F64, 1e-3).grade(), PrecisionGrade::Poor);
    assert_eq!(
        report(CanaryWidth::F32, 1e-7).grade(),
        PrecisionGrade::Exact
    );
    // f32 has no single-grade band: worse than exact is poor.
    assert_eq!(report(CanaryWidth::F32, 1e-4).grade(), PrecisionGrade::Poor);
}
