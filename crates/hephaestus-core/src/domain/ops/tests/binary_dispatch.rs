use super::*;

#[test]
fn combine_and_identity_agree_per_dialect() {
    assert_eq!(<SumOp as CombineExpr<Wgsl>>::EXPR, "lhs + rhs");
    assert_eq!(<SumOp as CombineExpr<CudaC>>::EXPR, "lhs + rhs");
    assert_eq!(<SumOp as CombineExpr<HipC>>::EXPR, "lhs + rhs");
    assert_eq!(<ProdOp as CombineExpr<Wgsl>>::EXPR, "lhs * rhs");
    assert_eq!(<ProdOp as CombineExpr<CudaC>>::EXPR, "lhs * rhs");
    assert_eq!(<ProdOp as CombineExpr<HipC>>::EXPR, "lhs * rhs");
    assert_eq!(<AddOp as BinaryExpr<HipC>>::EXPR, "lhs + rhs");
    assert_eq!(<NegOp as UnaryExpr<HipC>>::EXPR, "-x");
    assert_eq!(
        <GeluTanhOp as UnaryExpr<Wgsl>>::EXPR,
        "0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))"
    );
    assert!(<SiluGradOp as UnaryExpr<CudaC>>::EXPR.contains("exp(-x)"));
    assert!(<SoftplusOp as UnaryExpr<HipC>>::EXPR.contains("exp(x)"));
    assert!(<MishOp as UnaryExpr<Wgsl>>::EXPR.contains("tanh(log(1.0 + exp(x)))"));
    assert!(<MishGradOp as UnaryExpr<CudaC>>::EXPR.contains("tanhf(logf(1.0f + expf(x)))"));
    assert_eq!(
        <EluOp as UnaryExpr<HipC>>::EXPR,
        "x >= 0.0f ? x : expf(x) - 1.0f"
    );
    assert_eq!(
        <EluGradOp as UnaryExpr<Wgsl>>::EXPR,
        "select(exp(x), 1.0, x >= 0.0)"
    );
    assert_eq!(
        <HardsigmoidOp as UnaryExpr<HipC>>::EXPR,
        "fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f)"
    );
    assert_eq!(
        <HardsigmoidGradOp as UnaryExpr<HipC>>::EXPR,
        "(x > -3.0f && x < 3.0f) ? (1.0f / 6.0f) : 0.0f"
    );
    assert_eq!(
        <HardswishOp as UnaryExpr<HipC>>::EXPR,
        "x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f"
    );
    assert_eq!(
        <HardswishGradOp as UnaryExpr<HipC>>::EXPR,
        "x >= 3.0f ? 1.0f : (x > -3.0f ? (2.0f * x + 3.0f) / 6.0f : 0.0f)"
    );
    assert_eq!(
        <SoftsignOp as UnaryExpr<HipC>>::EXPR,
        "x / (1.0f + fabsf(x))"
    );
    assert_eq!(
        <SoftsignGradOp as UnaryExpr<HipC>>::EXPR,
        "1.0f / ((1.0f + fabsf(x)) * (1.0f + fabsf(x)))"
    );
    assert_eq!(
        <Log10Op as UnaryExpr<Wgsl>>::EXPR,
        "log(x) * 0.43429448190325182"
    );
    assert_eq!(<Expm1Op as UnaryExpr<CudaC>>::EXPR, "expm1(x)");
    assert_eq!(<Expm1Op as UnaryExpr<HipC>>::EXPR, "expm1(x)");
    assert_eq!(<Log1pOp as UnaryExpr<CudaC>>::EXPR, "log1p(x)");
    assert_eq!(<Log1pOp as UnaryExpr<HipC>>::EXPR, "log1p(x)");
    assert_eq!(<RoundOp as UnaryExpr<HipC>>::EXPR, "rint(x)");
    assert_eq!(<ErfOp as UnaryExpr<CudaC>>::EXPR, "erf(x)");
    assert_eq!(<ErfcOp as UnaryExpr<HipC>>::EXPR, "erfc(x)");
    assert_eq!(<LgammaOp as UnaryExpr<CudaC>>::EXPR, "lgamma(x)");
    assert_eq!(<LgammaOp as UnaryExpr<HipC>>::EXPR, "lgamma(x)");
    assert!(<ErfOp as UnaryExpr<Wgsl>>::EXPR.contains("1.061405429"));
    assert!(<ErfcOp as UnaryExpr<Wgsl>>::EXPR.starts_with("(1.0 - "));
    assert!(<LgammaOp as UnaryExpr<Wgsl>>::EXPR.contains("676.5203681218851"));
    assert!(<LgammaOp as UnaryExpr<Wgsl>>::EXPR.contains("abs(x) > 3.402823466e+38"));
    assert!(<GeluOp as UnaryExpr<Wgsl>>::EXPR.contains("0.7071067811865476"));
    assert!(<GeluGradOp as UnaryExpr<CudaC>>::EXPR.contains("erff(x *"));
    assert!(<GeluGradOp as UnaryExpr<HipC>>::EXPR.contains("expf(-0.5f * x * x)"));
    assert_eq!(
        <SignOp as UnaryExpr<Wgsl>>::EXPR,
        "select(select(0.0, -1.0, x < 0.0), 1.0, x > 0.0)"
    );
    assert_eq!(
        <SincOp as UnaryExpr<Wgsl>>::EXPR,
        "select(sin(x) / x, 1.0, x == 0.0)"
    );
    // KERNEL-TAIL: the WGSL tail forms branch — the expm1 small arm is
    // the degree-8 Taylor series at |x| < 0.5 (pin the 1/6 Horner
    // coefficient), the log1p small arm the degree-6 Chebyshev form at
    // |x| < 0.15 (pin its leading coefficient) — and both keep the naive
    // transcription as the outer arm.
    for expr in [
        <Expm1Op as UnaryExpr<Wgsl>>::EXPR,
        <Log1pOp as UnaryExpr<Wgsl>>::EXPR,
    ] {
        assert!(expr.starts_with("select("), "wgsl tail form branches");
        assert_eq!(
            expr.chars().filter(|&c| c == '(').count(),
            expr.chars().filter(|&c| c == ')').count(),
            "wgsl tail parens balance"
        );
    }
    assert!(<Expm1Op as UnaryExpr<Wgsl>>::EXPR.contains("abs(x) < 0.5"));
    assert!(<Expm1Op as UnaryExpr<Wgsl>>::EXPR.contains("0.16666666666666666"));
    assert!(<Expm1Op as UnaryExpr<Wgsl>>::EXPR.contains("exp(x) - 1.0"));
    assert!(<Log1pOp as UnaryExpr<Wgsl>>::EXPR.contains("abs(x) < 0.15"));
    assert!(<Log1pOp as UnaryExpr<Wgsl>>::EXPR.contains("1.00000010489199531"));
    assert_eq!(
        <SincOp as UnaryExpr<CudaC>>::EXPR,
        "((x) == 0.0f ? 1.0f : sin(x) / (x))"
    );
    assert_eq!(
        <SincOp as UnaryExpr<HipC>>::EXPR,
        "((x) == 0.0f ? 1.0f : sin(x) / (x))"
    );
    // J0/J1 spellings are ~2KB per dialect; pin the branch structure, one
    // coefficient per branch, and the Hankel trig, plus mechanical paren
    // balance (the strings are machine-built and driver-compiled).
    for expr in [
        <J0Op as UnaryExpr<Wgsl>>::EXPR,
        <J1Op as UnaryExpr<Wgsl>>::EXPR,
    ] {
        assert!(
            expr.starts_with("select("),
            "wgsl bessel branches on select"
        );
        assert!(
            expr.contains("abs(x) < 8.0"),
            "wgsl bessel crossover at |x| = 8"
        );
        assert_eq!(
            expr.chars().filter(|&c| c == '(').count(),
            expr.chars().filter(|&c| c == ')').count(),
            "wgsl bessel parens balance"
        );
    }
    assert!(<J0Op as UnaryExpr<Wgsl>>::EXPR.contains("57568490574.0"));
    assert!(<J0Op as UnaryExpr<Wgsl>>::EXPR.contains("0.7853981633974483"));
    assert!(<J0Op as UnaryExpr<Wgsl>>::EXPR.contains(", 1.0, x == 0.0)"));
    assert!(<J1Op as UnaryExpr<Wgsl>>::EXPR.contains("72362614232.0"));
    assert!(<J1Op as UnaryExpr<Wgsl>>::EXPR.contains("2.356194490192345"));
    assert!(<J1Op as UnaryExpr<Wgsl>>::EXPR.contains("* sign(x)"));
    for expr in [
        <J0Op as UnaryExpr<CudaC>>::EXPR,
        <J1Op as UnaryExpr<CudaC>>::EXPR,
        <J0Op as UnaryExpr<HipC>>::EXPR,
        <J1Op as UnaryExpr<HipC>>::EXPR,
    ] {
        assert!(
            expr.contains("fabs(x) < 8.0"),
            "c bessel crossover at |x| = 8"
        );
        assert!(
            !expr.contains("fabsf"),
            "c bessel keeps fabs unsuffixed for f64"
        );
        assert_eq!(
            expr.chars().filter(|&c| c == '(').count(),
            expr.chars().filter(|&c| c == ')').count(),
            "c bessel parens balance"
        );
    }
    assert_eq!(
        <J0Op as UnaryExpr<CudaC>>::EXPR,
        <J0Op as UnaryExpr<HipC>>::EXPR,
        "cuda and hip share one j0 spelling"
    );
    assert_eq!(
        <J1Op as UnaryExpr<CudaC>>::EXPR,
        <J1Op as UnaryExpr<HipC>>::EXPR,
        "cuda and hip share one j1 spelling"
    );
    // K0 pins the guard structure: `select` takes the false arm first, so
    // the (x - x)/(x - x) NaN arm leads and the value select trails the
    // domain check. One coefficient per branch, plus paren balance.
    assert!(
        <K0Op as UnaryExpr<Wgsl>>::EXPR.starts_with("select((((x) - (x)) / ((x) - (x))), select(")
    );
    assert!(<K0Op as UnaryExpr<Wgsl>>::EXPR.contains(", (x) <= 2.0), ((x) > 0.0)"));
    assert!(<K0Op as UnaryExpr<Wgsl>>::EXPR.contains("abs(x) <= 3.402823466e+38"));
    assert!(<K0Op as UnaryExpr<Wgsl>>::EXPR.contains("3.5156229"));
    assert!(<K0Op as UnaryExpr<Wgsl>>::EXPR.contains("1.25331414"));
    assert!(<K0Op as UnaryExpr<Wgsl>>::EXPR.contains("(((x) - (x)) / ((x) - (x)))"));
    assert_eq!(
        <K0Op as UnaryExpr<Wgsl>>::EXPR
            .chars()
            .filter(|&c| c == '(')
            .count(),
        <K0Op as UnaryExpr<Wgsl>>::EXPR
            .chars()
            .filter(|&c| c == ')')
            .count(),
        "wgsl k0 parens balance"
    );
    for expr in [
        <K0Op as UnaryExpr<CudaC>>::EXPR,
        <K0Op as UnaryExpr<HipC>>::EXPR,
    ] {
        assert!(expr.contains("isfinite(x)"), "c k0 guards on isfinite");
        assert!(expr.contains("(((x) - (x)) / ((x) - (x)))"), "c k0 nan arm");
        assert_eq!(
            expr.chars().filter(|&c| c == '(').count(),
            expr.chars().filter(|&c| c == ')').count(),
            "c k0 parens balance"
        );
    }
    assert_eq!(
        <K0Op as UnaryExpr<CudaC>>::EXPR,
        <K0Op as UnaryExpr<HipC>>::EXPR,
        "cuda and hip share one k0 spelling"
    );
    assert_eq!(<f32 as IdentityToken<SumOp, Wgsl>>::TOKEN, "0.0");
    assert_eq!(<f32 as IdentityToken<SumOp, CudaC>>::TOKEN, "0.0f");
    assert_eq!(<f32 as IdentityToken<SumOp, HipC>>::TOKEN, "0.0f");
    assert_eq!(<f32 as IdentityToken<ProdOp, Wgsl>>::TOKEN, "1.0");
    assert_eq!(<f32 as IdentityToken<ProdOp, CudaC>>::TOKEN, "1.0f");
    assert_eq!(<f32 as IdentityToken<ProdOp, HipC>>::TOKEN, "1.0f");
    assert_eq!(<u32 as IdentityToken<ProdOp, Wgsl>>::TOKEN, "1u");
    assert_eq!(<u32 as IdentityToken<ProdOp, CudaC>>::TOKEN, "1u");
    assert_eq!(<u32 as IdentityToken<ProdOp, HipC>>::TOKEN, "1u");
    assert_eq!(<i32 as IdentityToken<ProdOp, Wgsl>>::TOKEN, "1");
    assert_eq!(<i32 as IdentityToken<ProdOp, CudaC>>::TOKEN, "1");
    assert_eq!(<i32 as IdentityToken<ProdOp, HipC>>::TOKEN, "1");
    assert_eq!(<f32 as OpIdentity<MinOp>>::IDENTITY, f32::MAX);
    assert_eq!(<f32 as OpIdentity<ProdOp>>::IDENTITY, 1.0);
    assert_eq!(<u32 as OpIdentity<ProdOp>>::IDENTITY, 1);
    assert_eq!(<i32 as OpIdentity<ProdOp>>::IDENTITY, 1);
    assert_eq!(<u32 as OpIdentity<MaxOp>>::IDENTITY, u32::MIN);
}

#[test]
fn comparisons_use_scalar_correct_mask_literals() {
    assert_eq!(
        <EqOp as TypedBinaryExpr<Wgsl, f32>>::EXPR,
        "select(0.0, 1.0, lhs == rhs)"
    );
    assert_eq!(
        <EqOp as TypedBinaryExpr<Wgsl, u32>>::EXPR,
        "select(0u, 1u, lhs == rhs)"
    );
    assert_eq!(
        <EqOp as TypedBinaryExpr<Wgsl, i32>>::EXPR,
        "select(0, 1, lhs == rhs)"
    );
    assert_eq!(
        <EqOp as TypedBinaryExpr<Wgsl, f64>>::EXPR,
        "select(f64(0.0), f64(1.0), lhs == rhs)"
    );
    // Every f64 mask spelling carries explicit `f64(...)` arms: `select`
    // with AbstractFloat-only arms concretizes to `f32`, which naga rejects
    // at a typed `array<f64>` assignment.
    assert!(<NeOp as TypedBinaryExpr<Wgsl, f64>>::EXPR.contains("f64(0.0)"));
    assert!(<NeOp as TypedBinaryExpr<Wgsl, f64>>::EXPR.contains("lhs != rhs"));
    assert!(<LtOp as TypedBinaryExpr<Wgsl, f64>>::EXPR.contains("f64(1.0)"));
    assert!(<LtOp as TypedBinaryExpr<Wgsl, f64>>::EXPR.contains("lhs < rhs"));
    assert!(<GtOp as TypedBinaryExpr<Wgsl, f64>>::EXPR.contains("lhs > rhs"));
    assert!(<LeOp as TypedBinaryExpr<Wgsl, f64>>::EXPR.contains("lhs <= rhs"));
    assert!(<GeOp as TypedBinaryExpr<Wgsl, f64>>::EXPR.contains("lhs >= rhs"));
    // The literal-`select` unary spellings opt out of f64 while the
    // typed-arm majority keeps the default. Const-evaluated: a wrong flag
    // fails the build, not the suite.
    const _: () = assert!(!<SignOp as UnaryExpr<Wgsl>>::SUPPORTS_F64);
    const _: () = assert!(!<ReluGradOp as UnaryExpr<Wgsl>>::SUPPORTS_F64);
    const _: () = assert!(!<HardsigmoidGradOp as UnaryExpr<Wgsl>>::SUPPORTS_F64);
    const _: () = assert!(!<SincOp as UnaryExpr<Wgsl>>::SUPPORTS_F64);
    const _: () = assert!(!<J0Op as UnaryExpr<Wgsl>>::SUPPORTS_F64);
    const _: () = assert!(!<J1Op as UnaryExpr<Wgsl>>::SUPPORTS_F64);
    const _: () = assert!(<SignOp as UnaryExpr<CudaC>>::SUPPORTS_F64);
    const _: () = assert!(<SincOp as UnaryExpr<CudaC>>::SUPPORTS_F64);
    const _: () = assert!(<SincOp as UnaryExpr<HipC>>::SUPPORTS_F64);
    const _: () = assert!(<J0Op as UnaryExpr<CudaC>>::SUPPORTS_F64);
    const _: () = assert!(<J0Op as UnaryExpr<HipC>>::SUPPORTS_F64);
    const _: () = assert!(<J1Op as UnaryExpr<CudaC>>::SUPPORTS_F64);
    const _: () = assert!(<J1Op as UnaryExpr<HipC>>::SUPPORTS_F64);
    const _: () = assert!(!<K0Op as UnaryExpr<Wgsl>>::SUPPORTS_F64);
    const _: () = assert!(<K0Op as UnaryExpr<CudaC>>::SUPPORTS_F64);
    const _: () = assert!(<K0Op as UnaryExpr<HipC>>::SUPPORTS_F64);
    const _: () = assert!(<ReluOp as UnaryExpr<Wgsl>>::SUPPORTS_F64);
    const _: () = assert!(<Log10Op as UnaryExpr<Wgsl>>::SUPPORTS_F64);
    // KERNEL-TAIL: the WGSL tail forms nest `select` and opt out of f64
    // (the expm1 Taylor degree is f32-graded besides); the native C
    // builtins serve both precisions through lane-width overload.
    const _: () = assert!(!<Expm1Op as UnaryExpr<Wgsl>>::SUPPORTS_F64);
    const _: () = assert!(!<Log1pOp as UnaryExpr<Wgsl>>::SUPPORTS_F64);
    const _: () = assert!(<Expm1Op as UnaryExpr<CudaC>>::SUPPORTS_F64);
    const _: () = assert!(<Expm1Op as UnaryExpr<HipC>>::SUPPORTS_F64);
    const _: () = assert!(<Log1pOp as UnaryExpr<CudaC>>::SUPPORTS_F64);
    const _: () = assert!(<Log1pOp as UnaryExpr<HipC>>::SUPPORTS_F64);
    assert_eq!(
        <GeOp as TypedBinaryExpr<CudaC, f32>>::EXPR,
        "lhs >= rhs ? 1.0f : 0.0f"
    );
    assert_eq!(
        <EqOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs == rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <NeOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs != rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <LtOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs < rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <GtOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs > rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <LeOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs <= rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <GeOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs >= rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <GeOp as TypedBinaryExpr<CudaC, u32>>::EXPR,
        "lhs >= rhs ? 1u : 0u"
    );
    assert_eq!(
        <GeOp as TypedBinaryExpr<HipC, i32>>::EXPR,
        "lhs >= rhs ? 1 : 0"
    );
}

#[test]
fn consumer_defined_op_composes_with_the_vocabulary() {
    // A consumer-side fused op: implement the expression trait for a
    // local ZST in the targeted dialect — no substrate changes needed.
    #[derive(Clone, Copy, Debug, Default)]
    struct AffineClampOp;
    impl UnaryExpr<Wgsl> for AffineClampOp {
        const EXPR: &'static str = "clamp(x * 2.0 + 1.0, 0.0, 10.0)";
    }
    fn expr_of<Op: UnaryExpr<Wgsl>>() -> &'static str {
        Op::EXPR
    }
    assert_eq!(
        expr_of::<AffineClampOp>(),
        "clamp(x * 2.0 + 1.0, 0.0, 10.0)"
    );
}

/// Integer combines wrap (the WGSL kernel semantics); float combines are
/// IEEE; min/max keep `lhs` unless `rhs` is strictly smaller/larger.
#[test]
fn host_combines_apply_the_value_functions() {
    assert_eq!(
        <SumOp as CombineExpr<Host>>::value(i32::MAX, 1),
        Some(i32::MIN)
    );
    assert_eq!(
        <ProdOp as CombineExpr<Host>>::value(u32::MAX, 2),
        Some(u32::MAX - 1)
    );
    assert_eq!(
        <CumSumOp as CombineExpr<Host>>::value(1.5f32, 2.25),
        Some(3.75)
    );
    assert_eq!(<CumProdOp as CombineExpr<Host>>::value(-3i32, 4), Some(-12));
    assert_eq!(
        <MinOp as CombineExpr<Host>>::value(2.0f64, -1.0),
        Some(-1.0)
    );
    assert_eq!(<MaxOp as CombineExpr<Host>>::value(2u32, 7), Some(7));
    let kept = <MinOp as CombineExpr<Host>>::value(1.0f32, f32::NAN).expect("min has a value");
    assert_eq!(kept, 1.0, "a NaN rhs never displaces a held number");
    assert_eq!(<SumOp as CombineExpr<Host>>::EXPR, "host");
    assert_eq!(<f32 as IdentityToken<MinOp, Host>>::TOKEN, "host");
}

/// An operator implementing the host combine without a value function
/// reports `None`, which the host turns into a typed error; other dialects
/// never carry a value.
#[test]
fn a_host_combine_without_a_value_function_has_none() {
    #[derive(Clone, Copy)]
    struct Opaque;
    impl CombineExpr<Host> for Opaque {
        const EXPR: &'static str = "host";
    }
    assert_eq!(<Opaque as CombineExpr<Host>>::value(1.0f32, 2.0), None);
    assert_eq!(<SumOp as CombineExpr<Wgsl>>::value(1.0f32, 2.0), None);
}

/// Integer `Add`/`Sub`/`Mul` wrap; `Div` returns the dividend on a zero
/// divisor or `MIN / -1` (WGSL semantics, ADR 0061 Decision 5); float `Div`
/// is IEEE.
#[test]
fn host_binary_value_functions_match_the_operator_definition() {
    assert_eq!(
        <AddOp as BinaryExpr<Host>>::value(i32::MAX, 1),
        Some(i32::MIN)
    );
    assert_eq!(
        <SubOp as BinaryExpr<Host>>::value(i32::MIN, 1),
        Some(i32::MAX)
    );
    assert_eq!(
        <MulOp as BinaryExpr<Host>>::value(u32::MAX, 2),
        Some(u32::MAX - 1)
    );
    assert_eq!(<DivOp as BinaryExpr<Host>>::value(7i32, 0), Some(7));
    assert_eq!(
        <DivOp as BinaryExpr<Host>>::value(i32::MIN, -1),
        Some(i32::MIN)
    );
    assert_eq!(<DivOp as BinaryExpr<Host>>::value(7.0f32, 2.0), Some(3.5));
    assert_eq!(<AddOp as BinaryExpr<Host>>::EXPR, "host");
}

/// ADR 0061 Verification plan: `f32` `Add` reaches `apply` through the
/// `apply_real` default (Add implements only `apply`), and `Pow` computes
/// through `apply_real` while its `apply` (the `NumericElement`-generic path
/// every integer scalar dispatches through) reports `None` for both `f32`
/// and `i32` alike — the host's per-scalar dispatch (`hephaestus-host`) is
/// what routes `f32` to `real_value` and `i32` to `value`.
#[test]
fn binary_dispatch_matches_the_adr_0061_verification_plan() {
    assert_eq!(
        <AddOp as BinaryExpr<Host>>::real_value(2.0f32, 3.0),
        Some(5.0)
    );
    assert_eq!(<PowOp as BinaryExpr<Host>>::value::<f32>(2.0, 3.0), None);
    assert_eq!(<PowOp as BinaryExpr<Host>>::value::<i32>(2, 3), None);
    assert_eq!(
        <PowOp as BinaryExpr<Host>>::real_value(2.0f32, 3.0),
        Some(8.0)
    );
}

/// Comparisons produce the type's `1`/`0` indicator, generic over every
/// `NumericElement` scalar; NaN follows `PartialOrd`'s unordered contract.
#[test]
fn host_typed_binary_value_functions_produce_indicators() {
    assert_eq!(
        <EqOp as TypedBinaryExpr<Host, f32>>::value(1.0, 1.0),
        Some(1.0)
    );
    assert_eq!(
        <EqOp as TypedBinaryExpr<Host, f32>>::value(1.0, 2.0),
        Some(0.0)
    );
    assert_eq!(
        <NeOp as TypedBinaryExpr<Host, f32>>::value(f32::NAN, f32::NAN),
        Some(1.0)
    );
    assert_eq!(
        <LtOp as TypedBinaryExpr<Host, f32>>::value(f32::NAN, 1.0),
        Some(0.0)
    );
    assert_eq!(<LtOp as TypedBinaryExpr<Host, i32>>::value(-5, 2), Some(1));
    assert_eq!(<GeOp as TypedBinaryExpr<Host, u32>>::value(3, 3), Some(1));
    assert_eq!(<EqOp as TypedBinaryExpr<Host, f32>>::EXPR, "host");
}

/// Integer `Div` edge cases and `Add`/`Mul` wraparound, generic over `i32`
/// and `u32` (the seam's shipped signed/unsigned integers).
#[test]
fn integer_binary_edge_cases_hold_for_every_shipped_integer_scalar() {
    assert_eq!(<DivOp as BinaryExpr<Host>>::value(5i32, 0), Some(5));
    assert_eq!(
        <DivOp as BinaryExpr<Host>>::value(i32::MIN, -1),
        Some(i32::MIN)
    );
    assert_eq!(<DivOp as BinaryExpr<Host>>::value(5u32, 0), Some(5));
    assert_eq!(<DivOp as BinaryExpr<Host>>::value(7i32, 2), Some(3));
    assert_eq!(
        <AddOp as BinaryExpr<Host>>::value(i32::MAX, 1),
        Some(i32::MIN)
    );
    assert_eq!(<AddOp as BinaryExpr<Host>>::value(u32::MAX, 1), Some(0u32));
    assert_eq!(
        <MulOp as BinaryExpr<Host>>::value(i32::MAX, 2),
        Some(i32::MAX.wrapping_mul(2))
    );
    assert_eq!(
        <MulOp as BinaryExpr<Host>>::value(u32::MAX, 2),
        Some(u32::MAX.wrapping_mul(2))
    );
}
