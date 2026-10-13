use super::super::*;

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
    // KERNEL-TAIL: the WGSL softplus tree nests `select` over f32-graded
    // arms (and ADR 0072 refuses f64 `exp` on Vulkan); the native C
    // builtins serve both precisions.
    const _: () = assert!(!<SoftplusOp as UnaryExpr<Wgsl>>::SUPPORTS_F64);
    const _: () = assert!(<SoftplusOp as UnaryExpr<CudaC>>::SUPPORTS_F64);
    const _: () = assert!(<SoftplusOp as UnaryExpr<HipC>>::SUPPORTS_F64);
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
