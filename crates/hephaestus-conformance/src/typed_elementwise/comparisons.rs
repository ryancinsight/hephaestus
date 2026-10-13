use hephaestus_core::{
    ComputeDevice, DialectScalar, ElementwiseOps, EqOp, GeOp, GtOp, LeOp, LtOp, NeOp,
    TypedBinaryExpr,
};

use super::helpers::compare;

/// `u32` comparisons: unsigned ordering with an equal pair at both ends.
pub(crate) fn unsigned_comparisons_are_exact_indicators<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, u32>,
    u32: DialectScalar<E::Dialect>,
    EqOp: TypedBinaryExpr<E::Dialect, u32>,
    NeOp: TypedBinaryExpr<E::Dialect, u32>,
    LtOp: TypedBinaryExpr<E::Dialect, u32>,
    LeOp: TypedBinaryExpr<E::Dialect, u32>,
    GtOp: TypedBinaryExpr<E::Dialect, u32>,
    GeOp: TypedBinaryExpr<E::Dialect, u32>,
{
    let a: [u32; 5] = [0, 1, 7, 7, 4_294_967_295];
    let b: [u32; 5] = [0, 2, 7, 3, 0];
    let name = device.backend_name();
    assert_eq!(
        compare::<_, _, _, EqOp, 5>(device, ops, &a, &b),
        [1u32, 0, 1, 0, 0],
        "{name}: u32 eq"
    );
    assert_eq!(
        compare::<_, _, _, NeOp, 5>(device, ops, &a, &b),
        [0u32, 1, 0, 1, 1],
        "{name}: u32 ne"
    );
    assert_eq!(
        compare::<_, _, _, LtOp, 5>(device, ops, &a, &b),
        [0u32, 1, 0, 0, 0],
        "{name}: u32 lt"
    );
    assert_eq!(
        compare::<_, _, _, LeOp, 5>(device, ops, &a, &b),
        [1u32, 1, 1, 0, 0],
        "{name}: u32 le"
    );
    assert_eq!(
        compare::<_, _, _, GtOp, 5>(device, ops, &a, &b),
        [0u32, 0, 0, 1, 1],
        "{name}: u32 gt"
    );
    assert_eq!(
        compare::<_, _, _, GeOp, 5>(device, ops, &a, &b),
        [1u32, 0, 1, 1, 1],
        "{name}: u32 ge"
    );
}

/// `i32` comparisons: the discriminating case is negative operands, which a
/// kernel that reused the unsigned expression would order incorrectly.
pub(crate) fn signed_comparisons_order_by_sign<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, i32>,
    i32: DialectScalar<E::Dialect>,
    EqOp: TypedBinaryExpr<E::Dialect, i32>,
    LtOp: TypedBinaryExpr<E::Dialect, i32>,
    GtOp: TypedBinaryExpr<E::Dialect, i32>,
{
    let a: [i32; 5] = [-5, -1, 0, 3, i32::MIN];
    let b: [i32; 5] = [2, -1, -0, -3, i32::MAX];
    let name = device.backend_name();
    // -5 < 2, -1 !< -1, 0 !< 0, 3 !< -3, i32::MIN < i32::MAX.
    assert_eq!(
        compare::<_, _, _, LtOp, 5>(device, ops, &a, &b),
        [1i32, 0, 0, 0, 1],
        "{name}: i32 lt"
    );
    assert_eq!(
        compare::<_, _, _, GtOp, 5>(device, ops, &a, &b),
        [0i32, 0, 0, 1, 0],
        "{name}: i32 gt"
    );
    assert_eq!(
        compare::<_, _, _, EqOp, 5>(device, ops, &a, &b),
        [0i32, 1, 1, 0, 0],
        "{name}: i32 eq"
    );
}

/// `f32` comparisons over finite operands, including the signed-zero identity
/// `-0.0 == 0.0` and an adjacent pair at the `2^24` exact-integer limit.
pub(crate) fn finite_float_comparisons_are_exact_indicators<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    EqOp: TypedBinaryExpr<E::Dialect, f32>,
    NeOp: TypedBinaryExpr<E::Dialect, f32>,
    LtOp: TypedBinaryExpr<E::Dialect, f32>,
    GeOp: TypedBinaryExpr<E::Dialect, f32>,
{
    let a: [f32; 6] = [1.0, -0.0, 2.5, -3.0, 0.25, 16_777_216.0];
    let b: [f32; 6] = [1.0, 0.0, 2.0, -3.0, 0.5, 16_777_215.0];
    let name = device.backend_name();
    assert_eq!(
        compare::<_, _, _, EqOp, 6>(device, ops, &a, &b),
        [1.0f32, 1.0, 0.0, 1.0, 0.0, 0.0],
        "{name}: f32 eq"
    );
    assert_eq!(
        compare::<_, _, _, NeOp, 6>(device, ops, &a, &b),
        [0.0f32, 0.0, 1.0, 0.0, 1.0, 1.0],
        "{name}: f32 ne"
    );
    assert_eq!(
        compare::<_, _, _, LtOp, 6>(device, ops, &a, &b),
        [0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0],
        "{name}: f32 lt"
    );
    assert_eq!(
        compare::<_, _, _, GeOp, 6>(device, ops, &a, &b),
        [1.0f32, 1.0, 1.0, 1.0, 0.0, 1.0],
        "{name}: f32 ge"
    );
}
