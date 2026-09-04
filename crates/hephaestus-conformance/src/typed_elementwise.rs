//! Contract clauses for scalar-aware (typed) binary elementwise dispatch.
//!
//! Each clause is generic over the device and the elementwise seam, so every
//! backend runs the same assertions. Comparisons are the only operators with
//! per-scalar-type codegen, so `u32`, `i32`, and `f32` are each instantiated
//! rather than one standing in for the others, and every oracle is an exact
//! equality: comparisons produce indicator values, the float operands are
//! finite dyadic values (with the `-0.0 == 0.0` identity and an adjacent pair
//! at the `2^24` exact-integer limit), and the discriminating signed case is
//! negative operands, which a kernel reusing the unsigned expression would
//! order incorrectly. NaN and infinity operands are deliberately absent: WGSL
//! implementations may assume they do not occur, so their behaviour is
//! capability-gated, not a universal clause.

use eunomia::Pod;
use hephaestus_core::{
    AddOp, BinaryExpr, ComputeDevice, DialectScalar, ElementwiseOps, EqOp, GeOp, GtOp,
    KernelDialect, LeOp, LtOp, MulOp, NeOp, StridedView, TypedBinaryExpr,
};
use leto::Layout;

/// A backend able to run every clause in this module.
///
/// Bundles the scalar instantiations and the comparison-operator expressions
/// each dialect must define, so a backend satisfies one bound instead of the
/// full operator × scalar matrix at every clause.
pub trait TypedElementwiseBackend<D: ComputeDevice>:
    ElementwiseOps<D, u32> + ElementwiseOps<D, i32> + ElementwiseOps<D, f32>
where
    u32: DialectScalar<<Self as ElementwiseOps<D, u32>>::Dialect>,
    i32: DialectScalar<<Self as ElementwiseOps<D, i32>>::Dialect>,
    f32: DialectScalar<<Self as ElementwiseOps<D, f32>>::Dialect>,
    EqOp: TypedBinaryExpr<<Self as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Self as ElementwiseOps<D, i32>>::Dialect, i32>
        + TypedBinaryExpr<<Self as ElementwiseOps<D, f32>>::Dialect, f32>,
    NeOp: TypedBinaryExpr<<Self as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Self as ElementwiseOps<D, f32>>::Dialect, f32>,
    LtOp: TypedBinaryExpr<<Self as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Self as ElementwiseOps<D, i32>>::Dialect, i32>
        + TypedBinaryExpr<<Self as ElementwiseOps<D, f32>>::Dialect, f32>,
    LeOp: TypedBinaryExpr<<Self as ElementwiseOps<D, u32>>::Dialect, u32>,
    GtOp: TypedBinaryExpr<<Self as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Self as ElementwiseOps<D, i32>>::Dialect, i32>,
    GeOp: TypedBinaryExpr<<Self as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Self as ElementwiseOps<D, f32>>::Dialect, f32>,
{
}

impl<D, E> TypedElementwiseBackend<D> for E
where
    D: ComputeDevice,
    E: ElementwiseOps<D, u32> + ElementwiseOps<D, i32> + ElementwiseOps<D, f32>,
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
{
}

/// Dispatch one rank-1 typed comparison through the seam and return the
/// downloaded indicators.
fn compare<D, E, T, Op, const LEN: usize>(device: &D, ops: &E, lhs: &[T], rhs: &[T]) -> [T; LEN]
where
    D: ComputeDevice,
    E: ElementwiseOps<D, T>,
    T: Pod + DialectScalar<E::Dialect> + Default,
    Op: TypedBinaryExpr<E::Dialect, T>,
{
    let a = device.upload(lhs).expect("lhs upload");
    let b = device.upload(rhs).expect("rhs upload");
    let out = device.alloc_zeroed::<T>(LEN).expect("output alloc");
    let layout = Layout::c_contiguous([LEN]).expect("rank-1 layout");
    ops.typed_binary_into::<Op, 1>(
        device,
        StridedView::new(&a, &layout),
        StridedView::new(&b, &layout),
        StridedView::new(&out, &layout),
    )
    .expect("typed binary dispatch");
    let mut got = [T::default(); LEN];
    device.download(&out, &mut got).expect("download");
    got
}

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

/// `u32` comparisons: unsigned ordering with an equal pair at both ends.
fn unsigned_comparisons_are_exact_indicators<D, E>(device: &D, ops: &E)
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
fn signed_comparisons_order_by_sign<D, E>(device: &D, ops: &E)
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
fn finite_float_comparisons_are_exact_indicators<D, E>(device: &D, ops: &E)
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

/// The `_into` form writes caller-owned storage and repeated dispatch over
/// unchanged operands is idempotent, not accumulating.
fn into_form_is_idempotent_over_unchanged_operands<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, u32>,
    u32: DialectScalar<E::Dialect>,
    EqOp: TypedBinaryExpr<E::Dialect, u32>,
{
    let a = device.upload(&[7u32, 6, 5, 4, 3, 2]).expect("lhs upload");
    let b = device.upload(&[1u32, 6, 9, 4, 5, 2]).expect("rhs upload");
    let out = device.alloc_zeroed::<u32>(6).expect("output alloc");
    let layout = Layout::c_contiguous([6]).expect("rank-1 layout");
    let name = device.backend_name();

    for pass in ["first", "second"] {
        ops.typed_binary_into::<EqOp, 1>(
            device,
            StridedView::new(&a, &layout),
            StridedView::new(&b, &layout),
            StridedView::new(&out, &layout),
        )
        .expect("typed binary into");
        let mut got = [0u32; 6];
        device.download(&out, &mut got).expect("download");
        assert_eq!(got, [0, 1, 0, 1, 0, 1], "{name}: into-form {pass} dispatch");
    }
}

/// A prepared typed dispatch may be re-dispatched, and holds its operand
/// allocations rather than a snapshot of their contents.
fn prepared_dispatch_is_reusable_and_observes_writes<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, u32>,
    u32: DialectScalar<E::Dialect>,
    EqOp: TypedBinaryExpr<E::Dialect, u32>,
{
    let a = device.upload(&[1u32, 2, 3, 4]).expect("lhs upload");
    let b = device.upload(&[1u32, 9, 3, 9]).expect("rhs upload");
    let out = device.alloc_zeroed::<u32>(4).expect("output alloc");
    let layout = Layout::c_contiguous([4]).expect("rank-1 layout");
    let name = device.backend_name();

    let prepared = ops
        .prepare_typed_binary_into::<EqOp, 1>(
            device,
            StridedView::new(&a, &layout),
            StridedView::new(&b, &layout),
            StridedView::new(&out, &layout),
        )
        .expect("prepare typed binary");

    ops.dispatch_typed_binary::<1>(device, &prepared)
        .expect("first dispatch");
    let mut got = [0u32; 4];
    device.download(&out, &mut got).expect("first download");
    assert_eq!(got, [1, 0, 1, 0], "{name}: prepared typed first dispatch");

    // Re-dispatch is stable, not accumulating.
    ops.dispatch_typed_binary::<1>(device, &prepared)
        .expect("second dispatch");
    device.download(&out, &mut got).expect("second download");
    assert_eq!(
        got,
        [1, 0, 1, 0],
        "{name}: prepared typed dispatch must be idempotent over unchanged input"
    );

    // The plan holds the buffers: a later write is observed.
    device
        .write_buffer(&b, &[1u32, 2, 9, 4])
        .expect("rhs rewrite");
    ops.dispatch_typed_binary::<1>(device, &prepared)
        .expect("third dispatch");
    device.download(&out, &mut got).expect("third download");
    assert_eq!(
        got,
        [1, 1, 0, 1],
        "{name}: prepared typed dispatch must observe writes to its bound input"
    );
}

/// Operand shape disagreement is a typed rejection and the output is left
/// untouched, not truncated or partially written.
fn shape_mismatch_is_rejected_before_mutation<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, u32>,
    u32: DialectScalar<E::Dialect>,
    EqOp: TypedBinaryExpr<E::Dialect, u32>,
{
    let a = device.upload(&[1u32, 2, 3]).expect("lhs upload");
    let b = device.upload(&[1u32, 2]).expect("rhs upload");
    let out = device.upload(&[9u32, 9, 9]).expect("output upload");
    let three = Layout::c_contiguous([3]).expect("rank-1 layout");
    let two = Layout::c_contiguous([2]).expect("rank-1 layout");
    let name = device.backend_name();

    let result = ops.typed_binary_into::<EqOp, 1>(
        device,
        StridedView::new(&a, &three),
        StridedView::new(&b, &two),
        StridedView::new(&out, &three),
    );
    assert!(
        result.is_err(),
        "{name}: mismatched operand shapes must be rejected"
    );
    let mut got = [0u32; 3];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [9, 9, 9],
        "{name}: a rejected dispatch must not touch the output"
    );
}

/// Strided operands are read through their layouts rather than in buffer
/// order. A transposed view is the discriminating case: same bytes, different
/// traversal.
fn strided_operands_are_read_through_their_layouts<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, u32>,
    u32: DialectScalar<E::Dialect>,
    EqOp: TypedBinaryExpr<E::Dialect, u32>,
{
    // Buffer holds a 2x3 C-contiguous matrix [[1,2,3],[4,5,6]], read as its
    // 3x2 transpose [[1,4],[2,5],[3,6]] — same bytes, swapped strides.
    let source = device
        .upload(&[1u32, 2, 3, 4, 5, 6])
        .expect("source upload");
    let transposed = Layout::try_new([3, 2], [1, 3], 0).expect("valid conformance fixture layout");

    // Compare against the transpose's own values with two deliberate
    // mismatches.
    let probe = device.upload(&[1u32, 4, 9, 5, 3, 9]).expect("probe upload");
    let dense = Layout::c_contiguous([3, 2]).expect("dense layout");
    let out = device.alloc_zeroed::<u32>(6).expect("output alloc");
    let name = device.backend_name();

    ops.typed_binary_into::<EqOp, 2>(
        device,
        StridedView::new(&source, &transposed),
        StridedView::new(&probe, &dense),
        StridedView::new(&out, &dense),
    )
    .expect("strided typed binary");
    let mut got = [0u32; 6];
    device.download(&out, &mut got).expect("download");
    // Transpose reads [1,4,2,5,3,6]; probe is [1,4,9,5,3,9].
    assert_eq!(
        got,
        [1, 1, 0, 1, 1, 0],
        "{name}: strided operands must be read through their layouts"
    );
}

/// IEEE-754 `f32` special-value semantics, asserted only where the dialect
/// advertises them (`KernelDialect::IEEE_SPECIAL_VALUES`, ADR 0043).
///
/// A non-advertising dialect (WGSL) skips by construction through the const
/// branch below — its specification permits treating NaN and infinity as
/// absent, so no assertion about them can hold. Oracles are exact bit-class
/// checks: comparison indicators, `is_nan`, and signed-infinity equality.
fn special_values_follow_ieee_where_advertised<D, E>(device: &D, ops: &E)
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

/// Dispatch one rank-1 binary arithmetic op through the seam and return the
/// downloaded results.
fn arithmetic<D, E, Op, const LEN: usize>(
    device: &D,
    ops: &E,
    lhs: &[f32],
    rhs: &[f32],
) -> [f32; LEN]
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    Op: BinaryExpr<E::Dialect>,
{
    let a = device.upload(lhs).expect("lhs upload");
    let b = device.upload(rhs).expect("rhs upload");
    let out = device.alloc_zeroed::<f32>(LEN).expect("output alloc");
    let layout = Layout::c_contiguous([LEN]).expect("rank-1 layout");
    ops.binary_into::<Op, 1>(
        device,
        StridedView::new(&a, &layout),
        StridedView::new(&b, &layout),
        StridedView::new(&out, &layout),
    )
    .expect("binary dispatch");
    let mut got = [0.0f32; LEN];
    device.download(&out, &mut got).expect("download");
    got
}
