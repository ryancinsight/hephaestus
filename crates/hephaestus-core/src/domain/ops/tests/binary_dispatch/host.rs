use super::super::*;

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
