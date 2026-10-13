use hephaestus_core::{
    AddOp, BinaryExpr, ComputeDevice, DialectScalar, ElementwiseOps, EqOp, GeOp, GtOp, LeOp, LtOp,
    MulOp, NeOp, TypedBinaryExpr,
};

use super::backend::TypedElementwiseBackend;
use super::comparisons::{
    finite_float_comparisons_are_exact_indicators, signed_comparisons_order_by_sign,
    unsigned_comparisons_are_exact_indicators,
};
use super::reuse::{
    into_form_is_idempotent_over_unchanged_operands,
    prepared_dispatch_is_reusable_and_observes_writes,
};
use super::shapes::{
    shape_mismatch_is_rejected_before_mutation, strided_operands_are_read_through_their_layouts,
};
use super::special::special_values_follow_ieee_where_advertised;

/// Run every typed-elementwise clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_typed_elementwise_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: TypedElementwiseBackend<D>,
    u32: DialectScalar<<E as ElementwiseOps<D, u32>>::Dialect>,
    i32: DialectScalar<<E as ElementwiseOps<D, i32>>::Dialect>,
    f32: DialectScalar<<E as ElementwiseOps<D, f32>>::Dialect>,
    EqOp: TypedBinaryExpr<<E as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<E as ElementwiseOps<D, i32>>::Dialect, i32>
        + TypedBinaryExpr<<E as ElementwiseOps<D, f32>>::Dialect, f32>,
    NeOp: TypedBinaryExpr<<E as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<E as ElementwiseOps<D, f32>>::Dialect, f32>,
    LtOp: TypedBinaryExpr<<E as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<E as ElementwiseOps<D, i32>>::Dialect, i32>
        + TypedBinaryExpr<<E as ElementwiseOps<D, f32>>::Dialect, f32>,
    LeOp: TypedBinaryExpr<<E as ElementwiseOps<D, u32>>::Dialect, u32>,
    GtOp: TypedBinaryExpr<<E as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<E as ElementwiseOps<D, i32>>::Dialect, i32>,
    GeOp: TypedBinaryExpr<<E as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<E as ElementwiseOps<D, f32>>::Dialect, f32>,
    AddOp: BinaryExpr<<E as ElementwiseOps<D, f32>>::Dialect>,
    MulOp: BinaryExpr<<E as ElementwiseOps<D, f32>>::Dialect>,
{
    unsigned_comparisons_are_exact_indicators(device, ops);
    special_values_follow_ieee_where_advertised(device, ops);
    signed_comparisons_order_by_sign(device, ops);
    finite_float_comparisons_are_exact_indicators(device, ops);
    into_form_is_idempotent_over_unchanged_operands(device, ops);
    prepared_dispatch_is_reusable_and_observes_writes(device, ops);
    shape_mismatch_is_rejected_before_mutation(device, ops);
    strided_operands_are_read_through_their_layouts(device, ops);
}
