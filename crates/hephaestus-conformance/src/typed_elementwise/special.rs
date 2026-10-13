use hephaestus_core::{
    AddOp, BinaryExpr, ComputeDevice, DialectScalar, ElementwiseOps, EqOp, KernelDialect, LtOp,
    MulOp, NeOp, TypedBinaryExpr,
};

use super::helpers::arithmetic;
use super::helpers::compare;

/// IEEE-754 `f32` special-value semantics, asserted only where the dialect
/// advertises them (`KernelDialect::IEEE_SPECIAL_VALUES`, ADR 0043).
///
/// A non-advertising dialect (WGSL) skips by construction through the const
/// branch below — its specification permits treating NaN and infinity as
/// absent, so no assertion about them can hold. Oracles are exact bit-class
/// checks: comparison indicators, `is_nan`, and signed-infinity equality.
pub(crate) fn special_values_follow_ieee_where_advertised<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    EqOp: TypedBinaryExpr<E::Dialect, f32>,
    NeOp: TypedBinaryExpr<E::Dialect, f32>,
    LtOp: TypedBinaryExpr<E::Dialect, f32>,
    AddOp: BinaryExpr<E::Dialect>,
    MulOp: BinaryExpr<E::Dialect>,
{
    if !<E::Dialect as KernelDialect>::IEEE_SPECIAL_VALUES {
        return;
    }
    let name = device.backend_name();
    let nan = f32::NAN;
    let inf = f32::INFINITY;

    // Unordered comparisons: every ordered comparison involving NaN is
    // false; inequality is true.
    let lhs = [nan, nan, 1.0, nan];
    let rhs = [nan, 1.0, nan, 2.0];
    assert_eq!(
        compare::<_, _, f32, EqOp, 4>(device, ops, &lhs, &rhs),
        [0.0; 4],
        "{name}: NaN must compare unequal to everything, itself included"
    );
    assert_eq!(
        compare::<_, _, f32, NeOp, 4>(device, ops, &lhs, &rhs),
        [1.0; 4],
        "{name}: NaN != x must hold for every x"
    );
    assert_eq!(
        compare::<_, _, f32, LtOp, 4>(device, ops, &lhs, &rhs),
        [0.0; 4],
        "{name}: ordered comparison with NaN must be false"
    );

    // Propagation and directed infinities through addition, including the
    // indeterminate form Inf + (-Inf) -> NaN.
    let sums =
        arithmetic::<_, _, AddOp, 4>(device, ops, &[nan, inf, -inf, inf], &[1.0, 1.0, 1.0, -inf]);
    assert!(
        sums[0].is_nan(),
        "{name}: NaN + 1 must be NaN, got {}",
        sums[0]
    );
    assert_eq!(sums[1], inf, "{name}: Inf + 1 must stay +Inf");
    assert_eq!(sums[2], -inf, "{name}: -Inf + 1 must stay -Inf");
    assert!(
        sums[3].is_nan(),
        "{name}: Inf + (-Inf) must be NaN, got {}",
        sums[3]
    );

    // Propagation and sign handling through multiplication, including the
    // indeterminate form 0 * Inf -> NaN.
    let products =
        arithmetic::<_, _, MulOp, 4>(device, ops, &[nan, 0.0, inf, -inf], &[1.0, inf, 2.0, 3.0]);
    assert!(
        products[0].is_nan(),
        "{name}: NaN * 1 must be NaN, got {}",
        products[0]
    );
    assert!(
        products[1].is_nan(),
        "{name}: 0 * Inf must be NaN, got {}",
        products[1]
    );
    assert_eq!(products[2], inf, "{name}: Inf * 2 must stay +Inf");
    assert_eq!(products[3], -inf, "{name}: -Inf * 3 must stay -Inf");
}
