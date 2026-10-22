use super::super::*;

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
        "select(x / (1.0 + exp(-2.0 * (0.7978845608 * (x + 0.044715 * x * x * x)))), -0.0, (x) < -3.402823466e+38)"
    );
    assert_eq!(
        <GeluTanhOp as UnaryExpr<CudaC>>::EXPR,
        "(x) < -3.402823466e+38f ? -0.0f : (x) * ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))))"
    );
    assert_eq!(
        <GeluTanhOp as UnaryExpr<HipC>>::EXPR,
        "(x) < -3.402823466e+38f ? -0.0f : (x) * ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))))"
    );
    assert!(<SiluGradOp as UnaryExpr<CudaC>>::EXPR.contains("2.63f"));
    assert_eq!(
        <SoftplusOp as UnaryExpr<CudaC>>::EXPR,
        "((x) > 0.0f ? (x) : 0.0f) + log1p(exp(-fabs(x)))"
    );
    assert_eq!(
        <SoftplusOp as UnaryExpr<HipC>>::EXPR,
        "((x) > 0.0f ? (x) : 0.0f) + log1p(exp(-fabs(x)))"
    );
    assert_eq!(
        <MishOp as UnaryExpr<CudaC>>::EXPR,
        "(x) < -3.402823466e+38f ? -0.0f : (x) * tanh(((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x))))))"
    );
    assert_eq!(
        <MishOp as UnaryExpr<HipC>>::EXPR,
        "(x) < -3.402823466e+38f ? -0.0f : (x) * tanh(((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x))))))"
    );
    assert!(<MishOp as UnaryExpr<Wgsl>>::EXPR.contains("x * tanh(max(x, 0.0) +"));
    assert!(<MishOp as UnaryExpr<Wgsl>>::EXPR.contains("x > -14.5)"));
    assert!(<MishOp as UnaryExpr<Wgsl>>::EXPR.contains("x > -2.75)"));
    // The middle arm composes the second-order `t2` (pin the `2/15`
    // literal): the first-order gap exceeds the bound past `-3.12`.
    assert!(<MishOp as UnaryExpr<Wgsl>>::EXPR.contains("0.13333334"));
    assert!(<MishOp as UnaryExpr<Wgsl>>::EXPR.contains(", x, x > 9.0)"));
    assert_eq!(
        <MishGradOp as UnaryExpr<CudaC>>::EXPR,
        "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : tanh(((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x)))))) + (x) * ((((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x)))))) <= 1.55f ? 1.0f - tanh(((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x)))))) * tanh(((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x)))))) : ((4.0f * exp(-2.0f * ((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x))))))) / ((1.0f + exp(-2.0f * ((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x))))))) * (1.0f + exp(-2.0f * ((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x)))))))))) * (((x)) <= 0.0f ? exp(((x))) / (1.0f + exp(((x)))) : 1.0f / (1.0f + exp(-((x))))))"
    );
    assert_eq!(
        <MishGradOp as UnaryExpr<HipC>>::EXPR,
        "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : tanh(((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x)))))) + (x) * ((((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x)))))) <= 1.55f ? 1.0f - tanh(((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x)))))) * tanh(((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x)))))) : ((4.0f * exp(-2.0f * ((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x))))))) / ((1.0f + exp(-2.0f * ((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x))))))) * (1.0f + exp(-2.0f * ((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x)))))))))) * (((x)) <= 0.0f ? exp(((x))) / (1.0f + exp(((x)))) : 1.0f / (1.0f + exp(-((x))))))"
    );
    assert!(<MishGradOp as UnaryExpr<Wgsl>>::EXPR.contains("exp(x) * (1.0 + x)"));
    assert!(<MishGradOp as UnaryExpr<Wgsl>>::EXPR.contains("x > -2.75)"));
    assert!(<MishGradOp as UnaryExpr<Wgsl>>::EXPR.contains("0.13333334"));
    assert!(<MishGradOp as UnaryExpr<Wgsl>>::EXPR.contains("x > -14.5)"));
    // The middle-arm `(1 - sp^2)` parenthesizes the fragment: it is
    // `max + select(...)`, only `+`-safe, and the bare form under `*`
    // re-associates (`1 - max + sel*max + sel`, the `-5.46e-3` defect).
    assert!(<MishGradOp as UnaryExpr<Wgsl>>::EXPR.contains("(1.0 - ("));
    // The middle-arm `(1-sp²)` carries the `2S⁴/3` correction (pin the
    // `/1.5` divisor): the uncorrected gap peaks at `19` ULP at `-3.3`.
    assert!(<MishGradOp as UnaryExpr<Wgsl>>::EXPR.contains("/ 1.5)"));
    assert_eq!(
        <EluOp as UnaryExpr<CudaC>>::EXPR,
        "((x) >= 0.0f ? (x) : expm1(x))"
    );
    assert_eq!(
        <EluOp as UnaryExpr<HipC>>::EXPR,
        "((x) >= 0.0f ? (x) : expm1(x))"
    );
    assert!(<EluOp as UnaryExpr<Wgsl>>::EXPR.contains("abs(x) < 0.5"));
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
    assert!(<ErfcOp as UnaryExpr<Wgsl>>::EXPR.starts_with("select((2.0 - "));
    assert!(<LgammaOp as UnaryExpr<Wgsl>>::EXPR.contains("676.5203681218851"));
    assert!(<LgammaOp as UnaryExpr<Wgsl>>::EXPR.contains("abs(x) > 3.402823466e+38"));
    assert!(<GeluOp as UnaryExpr<Wgsl>>::EXPR.contains("0.7071067811865476"));
    assert_eq!(
        <GeluOp as UnaryExpr<CudaC>>::EXPR,
        "(x) < -3.402823466e+38f ? -0.0f : 0.5f * x * erfc(-x * 0.7071067811865476f)"
    );
    assert_eq!(
        <GeluOp as UnaryExpr<HipC>>::EXPR,
        "(x) < -3.402823466e+38f ? -0.0f : 0.5f * x * erfc(-x * 0.7071067811865476f)"
    );
    assert!(<GeluOp as UnaryExpr<Wgsl>>::EXPR.contains("1.26551223"));
    assert!(<GeluOp as UnaryExpr<Wgsl>>::EXPR.contains("(x) < -3.402823466e+38"));
    assert_eq!(
        <GeluGradOp as UnaryExpr<CudaC>>::EXPR,
        "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : 0.5f * erfc(-x * 0.7071067811865476f) + x * exp(-0.5f * x * x) * 0.3989422804014327f)"
    );
    assert_eq!(
        <GeluGradOp as UnaryExpr<HipC>>::EXPR,
        "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : 0.5f * erfc(-x * 0.7071067811865476f) + x * exp(-0.5f * x * x) * 0.3989422804014327f)"
    );
    assert!(<GeluGradOp as UnaryExpr<Wgsl>>::EXPR.contains("1.26551223"));
    assert!(<GeluGradOp as UnaryExpr<Wgsl>>::EXPR.contains("(x) > 3.402823466e+38"));
    assert_eq!(
        <GeluTanhGradOp as UnaryExpr<Wgsl>>::EXPR,
        "select((select((exp(-abs(2.0 * (0.7978845608 * (select(x, 0.0, abs(x) >= 40.0) + 0.044715 * select(x, 0.0, abs(x) >= 40.0) * select(x, 0.0, abs(x) >= 40.0) * select(x, 0.0, abs(x) >= 40.0)))))) / (1.0 + exp(-abs(2.0 * (0.7978845608 * (select(x, 0.0, abs(x) >= 40.0) + 0.044715 * select(x, 0.0, abs(x) >= 40.0) * select(x, 0.0, abs(x) >= 40.0) * select(x, 0.0, abs(x) >= 40.0)))))), 1.0 / (1.0 + exp(-abs(2.0 * (0.7978845608 * (select(x, 0.0, abs(x) >= 40.0) + 0.044715 * select(x, 0.0, abs(x) >= 40.0) * select(x, 0.0, abs(x) >= 40.0) * select(x, 0.0, abs(x) >= 40.0)))))), x >= 0.0) + (2.0 * select(x, 0.0, abs(x) >= 40.0)) * ((exp(-abs(2.0 * (0.7978845608 * (select(x, 0.0, abs(x) >= 40.0) + 0.044715 * select(x, 0.0, abs(x) >= 40.0) * select(x, 0.0, abs(x) >= 40.0) * select(x, 0.0, abs(x) >= 40.0)))))) / ((1.0 + exp(-abs(2.0 * (0.7978845608 * (select(x, 0.0, abs(x) >= 40.0) + 0.044715 * select(x, 0.0, abs(x) >= 40.0) * select(x, 0.0, abs(x) >= 40.0) * select(x, 0.0, abs(x) >= 40.0)))))) * (1.0 + exp(-abs(2.0 * (0.7978845608 * (select(x, 0.0, abs(x) >= 40.0) + 0.044715 * select(x, 0.0, abs(x) >= 40.0) * select(x, 0.0, abs(x) >= 40.0) * select(x, 0.0, abs(x) >= 40.0)))))))) * 0.7978845608 * (1.0 + 0.134145 * select(x, 0.0, abs(x) >= 40.0) * select(x, 0.0, abs(x) >= 40.0))), select(select(x, 0.0, abs(x) >= 40.0), 1.0, x >= 0.0), abs(x) >= 40.0)"
    );
    assert_eq!(
        <GeluTanhGradOp as UnaryExpr<CudaC>>::EXPR,
        "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 2.47f ? (((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) * (1.0f - ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))))) == 0.0f ? ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) : ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) + 2.0f * (x) * ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) * (1.0f - ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))))) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * (x) * (x))) : ((((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) * (exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))))) == 0.0f ? ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) : ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) + 2.0f * (x) * (((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) * (exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))))) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * (x) * (x)))))"
    );
    assert_eq!(
        <GeluTanhGradOp as UnaryExpr<HipC>>::EXPR,
        "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 2.47f ? (((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) * (1.0f - ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))))) == 0.0f ? ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) : ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) + 2.0f * (x) * ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) * (1.0f - ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))))) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * (x) * (x))) : ((((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) * (exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))))) == 0.0f ? ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) : ((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) + 2.0f * (x) * (((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))) <= 0.0f ? exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp((((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))) : 1.0f / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))))) * (exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x))))) / (1.0f + exp(-(((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))))))) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * (x) * (x)))))"
    );
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
    // coefficient), log1p is musl's `log1pf` core in four bands — and the
    // forms keep a direct transcription as an outer arm.
    for expr in [
        <Expm1Op as UnaryExpr<Wgsl>>::EXPR,
        <Log1pOp as UnaryExpr<Wgsl>>::EXPR,
        <MishOp as UnaryExpr<Wgsl>>::EXPR,
        <MishGradOp as UnaryExpr<Wgsl>>::EXPR,
        <GeluGradOp as UnaryExpr<Wgsl>>::EXPR,
    ] {
        assert!(expr.starts_with("select("), "wgsl tail form branches");
        assert_eq!(
            expr.chars().filter(|&c| c == '(').count(),
            expr.chars().filter(|&c| c == ')').count(),
            "wgsl tail parens balance"
        );
    }
    // KERNEL-TAIL grads: the C-dialect spellings are long single literals
    // with no `select` structure to eyeball, so every one pins paren
    // balance here -- an extra `)` only surfaces as an nvrtc failure
    // otherwise (the `MishGradOp` `))))))` defect).
    for expr in [
        <MishGradOp as UnaryExpr<CudaC>>::EXPR,
        <MishGradOp as UnaryExpr<HipC>>::EXPR,
        <GeluGradOp as UnaryExpr<CudaC>>::EXPR,
        <GeluGradOp as UnaryExpr<HipC>>::EXPR,
        <GeluTanhGradOp as UnaryExpr<CudaC>>::EXPR,
        <GeluTanhGradOp as UnaryExpr<HipC>>::EXPR,
    ] {
        assert_eq!(
            expr.chars().filter(|&c| c == '(').count(),
            expr.chars().filter(|&c| c == ')').count(),
            "c tail parens balance"
        );
    }
    assert!(<Expm1Op as UnaryExpr<Wgsl>>::EXPR.contains("abs(x) < 0.5"));
    assert!(<Expm1Op as UnaryExpr<Wgsl>>::EXPR.contains("0.16666666666666666"));
    assert!(<Expm1Op as UnaryExpr<Wgsl>>::EXPR.contains("exp(x) - 1.0"));
    assert!(<Log1pOp as UnaryExpr<Wgsl>>::EXPR.contains("(x) >= -0.2928932188134525"));
    assert!(<Log1pOp as UnaryExpr<Wgsl>>::EXPR.contains("0.40000972152"));
    // KERNEL-TAIL: the WGSL softplus tree is `max + C(-|x|)` with three
    // nested Chebyshev arms (pin each crossover plus the log2 leading
    // coefficient of the near arm and the ~1.0 leading coefficient of
    // the far arm).
    let softplus_wgsl = <SoftplusOp as UnaryExpr<Wgsl>>::EXPR;
    assert!(softplus_wgsl.starts_with("max(x, 0.0) + select("));
    assert!(softplus_wgsl.contains("(-abs(x)) > -2.0"));
    assert!(softplus_wgsl.contains("(-abs(x)) > -5.0"));
    assert!(softplus_wgsl.contains("(-abs(x)) > -14.5"));
    assert!(softplus_wgsl.contains("6.93147180559945397e-1"));
    assert!(softplus_wgsl.contains("9.99970853800730142e-1"));
    assert_eq!(
        softplus_wgsl.chars().filter(|&c| c == '(').count(),
        softplus_wgsl.chars().filter(|&c| c == ')').count(),
        "wgsl softplus parens balance"
    );
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
