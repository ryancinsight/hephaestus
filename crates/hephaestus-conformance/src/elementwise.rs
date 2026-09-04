//! Contract clauses for untyped unary and binary elementwise dispatch.
//!
//! Complements [`crate::typed_elementwise`], which owns the scalar-aware
//! comparison paths: these clauses cover the plain arithmetic surface —
//! `unary_into`, `binary_into`, their prepared forms, and strided traversal.
//!
//! Every fixture is dyadic (representable as `m·2^e`) and every operation
//! either preserves dyadic values exactly (negation, absolute value,
//! addition, subtraction, multiplication, division by powers of two) or is
//! IEEE-754 correctly rounded onto an exactly representable result
//! (`sqrt` of a perfect square). All oracles are therefore exact
//! equalities; no tolerance is involved.

use eunomia::Pod;
use hephaestus_core::{
    AbsOp, AddOp, BinaryExpr, ComputeDevice, DialectScalar, DivOp, ElementwiseOps, MulOp, NegOp,
    SqrtOp, StridedView, SubOp, UnaryExpr,
};
use leto::Layout;

/// Run every untyped elementwise clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_elementwise_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect> + Pod,
    NegOp: UnaryExpr<E::Dialect>,
    AbsOp: UnaryExpr<E::Dialect>,
    SqrtOp: UnaryExpr<E::Dialect>,
    AddOp: BinaryExpr<E::Dialect>,
    SubOp: BinaryExpr<E::Dialect>,
    MulOp: BinaryExpr<E::Dialect>,
    DivOp: BinaryExpr<E::Dialect>,
{
    unary_ops_compute_exact_values(device, ops);
    binary_ops_compute_exact_values(device, ops);
    strided_operands_are_read_through_their_layouts(device, ops);
    prepared_unary_rebinds_bound_operands(device, ops);
    prepared_binary_rebinds_bound_operands(device, ops);
    scalar_ops_compute_exact_values(device, ops);
    prepared_scalar_rebinds_bound_operands(device, ops);
    shape_mismatch_is_rejected_before_mutation(device, ops);
}

/// Dispatch one rank-1 unary op and return the downloaded results.
fn unary<D, E, Op, const LEN: usize>(device: &D, ops: &E, input: &[f32]) -> [f32; LEN]
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    Op: UnaryExpr<E::Dialect>,
{
    let a = device.upload(input).expect("input upload");
    let out = device.alloc_zeroed::<f32>(LEN).expect("output alloc");
    let layout = Layout::c_contiguous([LEN]).expect("rank-1 layout");
    ops.unary_into::<Op, 1>(
        device,
        StridedView::new(&a, &layout),
        StridedView::new(&out, &layout),
    )
    .expect("unary dispatch");
    let mut got = [0.0f32; LEN];
    device.download(&out, &mut got).expect("download");
    got
}

/// Dispatch one rank-1 binary op and return the downloaded results.
fn binary<D, E, Op, const LEN: usize>(device: &D, ops: &E, lhs: &[f32], rhs: &[f32]) -> [f32; LEN]
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

/// Negation and absolute value are sign flips (exact); square roots of
/// perfect squares are correctly rounded onto exact dyadic results.
fn unary_ops_compute_exact_values<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    NegOp: UnaryExpr<E::Dialect>,
    AbsOp: UnaryExpr<E::Dialect>,
    SqrtOp: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let signed = [1.5f32, -2.0, 0.0, -0.25];
    assert_eq!(
        unary::<_, _, NegOp, 4>(device, ops, &signed),
        [-1.5, 2.0, 0.0, 0.25],
        "{name}: negation"
    );
    assert_eq!(
        unary::<_, _, AbsOp, 4>(device, ops, &signed),
        [1.5, 2.0, 0.0, 0.25],
        "{name}: absolute value"
    );
    assert_eq!(
        unary::<_, _, SqrtOp, 4>(device, ops, &[4.0, 9.0, 0.25, 1.0]),
        [2.0, 3.0, 0.5, 1.0],
        "{name}: square root of perfect squares"
    );
}

/// Dyadic binary arithmetic: every result is exactly representable.
fn binary_ops_compute_exact_values<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    AddOp: BinaryExpr<E::Dialect>,
    SubOp: BinaryExpr<E::Dialect>,
    MulOp: BinaryExpr<E::Dialect>,
    DivOp: BinaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let lhs = [1.5f32, -2.0, 8.0, 0.25];
    let rhs = [0.5f32, 4.0, -2.0, 2.0];
    assert_eq!(
        binary::<_, _, AddOp, 4>(device, ops, &lhs, &rhs),
        [2.0, 2.0, 6.0, 2.25],
        "{name}: addition"
    );
    assert_eq!(
        binary::<_, _, SubOp, 4>(device, ops, &lhs, &rhs),
        [1.0, -6.0, 10.0, -1.75],
        "{name}: subtraction"
    );
    assert_eq!(
        binary::<_, _, MulOp, 4>(device, ops, &lhs, &rhs),
        [0.75, -8.0, -16.0, 0.5],
        "{name}: multiplication"
    );
    assert_eq!(
        binary::<_, _, DivOp, 4>(device, ops, &lhs, &rhs),
        [3.0, -0.5, -4.0, 0.125],
        "{name}: division by powers of two"
    );
}

/// A transposed input view is traversed through its strides, not its
/// storage order.
fn strided_operands_are_read_through_their_layouts<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    NegOp: UnaryExpr<E::Dialect>,
    AddOp: BinaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    // Storage [1,2,3,4] viewed as the transpose of the row-major 2x2, so the
    // logical matrix is [[1,3],[2,4]].
    let a = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).expect("upload");
    let transposed = Layout::try_new([2, 2], [1, 2], 0).expect("valid conformance fixture layout");
    let dense = Layout::c_contiguous([2, 2]).expect("dense layout");

    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    ops.unary_into::<NegOp, 2>(
        device,
        StridedView::new(&a, &transposed),
        StridedView::new(&out, &dense),
    )
    .expect("strided unary dispatch");
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [-1.0, -3.0, -2.0, -4.0],
        "{name}: unary must traverse the transposed layout"
    );

    let b = device.upload(&[10.0f32, 20.0, 30.0, 40.0]).expect("upload");
    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    ops.binary_into::<AddOp, 2>(
        device,
        StridedView::new(&a, &transposed),
        StridedView::new(&b, &dense),
        StridedView::new(&out, &dense),
    )
    .expect("strided binary dispatch");
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [11.0, 23.0, 32.0, 44.0],
        "{name}: binary must traverse the transposed lhs layout"
    );
}

/// A prepared unary dispatch re-runs over its bound operands: idempotent
/// over unchanged inputs, and writes to a bound input are observed.
fn prepared_unary_rebinds_bound_operands<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    NegOp: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let a = device.upload(&[1.0f32, -2.0, 3.0, -4.0]).expect("upload");
    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    let layout = Layout::c_contiguous([4]).expect("rank-1 layout");
    let prepared = ops
        .prepare_unary_into::<NegOp, 1>(
            device,
            StridedView::new(&a, &layout),
            StridedView::new(&out, &layout),
        )
        .expect("prepare unary");

    let expected = [-1.0f32, 2.0, -3.0, 4.0];
    ops.dispatch_unary::<1>(device, &prepared)
        .expect("dispatch");
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(got, expected, "{name}: prepared negation");

    ops.dispatch_unary::<1>(device, &prepared)
        .expect("re-dispatch");
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: prepared unary must be idempotent over unchanged operands"
    );

    device
        .write_buffer(&a, &[8.0f32, -8.0, 0.5, -0.5])
        .expect("input rewrite");
    ops.dispatch_unary::<1>(device, &prepared)
        .expect("rebind dispatch");
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [-8.0, 8.0, -0.5, 0.5],
        "{name}: prepared unary must observe writes to its bound operand"
    );
}

/// A prepared binary dispatch re-runs over its bound operands likewise.
fn prepared_binary_rebinds_bound_operands<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    AddOp: BinaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let a = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).expect("lhs upload");
    let b = device.upload(&[0.5f32, 0.5, 0.5, 0.5]).expect("rhs upload");
    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    let layout = Layout::c_contiguous([4]).expect("rank-1 layout");
    let prepared = ops
        .prepare_binary_into::<AddOp, 1>(
            device,
            StridedView::new(&a, &layout),
            StridedView::new(&b, &layout),
            StridedView::new(&out, &layout),
        )
        .expect("prepare binary");

    ops.dispatch_binary::<1>(device, &prepared)
        .expect("dispatch");
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(got, [1.5, 2.5, 3.5, 4.5], "{name}: prepared addition");

    device
        .write_buffer(&b, &[2.0f32, 2.0, 2.0, 2.0])
        .expect("rhs rewrite");
    ops.dispatch_binary::<1>(device, &prepared)
        .expect("rebind dispatch");
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [3.0, 4.0, 5.0, 6.0],
        "{name}: prepared binary must observe writes to its bound operands"
    );
}

/// A shape mismatch is rejected before any output element is written.
fn shape_mismatch_is_rejected_before_mutation<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    AddOp: BinaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let a = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).expect("lhs upload");
    let b = device.upload(&[1.0f32, 2.0]).expect("rhs upload");
    let out = device.upload(&[7.0f32, 7.0, 7.0, 7.0]).expect("sentinel");
    let four = Layout::c_contiguous([4]).expect("layout");
    let two = Layout::c_contiguous([2]).expect("layout");
    assert!(
        ops.binary_into::<AddOp, 1>(
            device,
            StridedView::new(&a, &four),
            StridedView::new(&b, &two),
            StridedView::new(&out, &four),
        )
        .is_err(),
        "{name}: shape mismatch must be rejected"
    );
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got, [7.0; 4],
        "{name}: rejected dispatch must not mutate the output"
    );
}

/// Broadcast-scalar arithmetic: dyadic operands and scalars, exact results.
fn scalar_ops_compute_exact_values<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    SubOp: BinaryExpr<E::Dialect>,
    MulOp: BinaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let input = [3.0f32, 0.5, -1.0, 8.0];
    let a = device.upload(&input).expect("input upload");
    let layout = Layout::c_contiguous([4]).expect("rank-1 layout");

    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    ops.scalar_into::<SubOp, 1>(
        device,
        StridedView::new(&a, &layout),
        2.0,
        StridedView::new(&out, &layout),
    )
    .expect("scalar sub dispatch");
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [1.0, -1.5, -3.0, 6.0],
        "{name}: broadcast-scalar subtraction"
    );

    ops.scalar_into::<MulOp, 1>(
        device,
        StridedView::new(&a, &layout),
        0.5,
        StridedView::new(&out, &layout),
    )
    .expect("scalar mul dispatch");
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [1.5, 0.25, -0.5, 4.0],
        "{name}: broadcast-scalar multiplication"
    );
}

/// A prepared broadcast-scalar dispatch re-runs over its bound buffers; the
/// scalar itself is fixed at preparation.
fn prepared_scalar_rebinds_bound_operands<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    AddOp: BinaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let a = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).expect("upload");
    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    let layout = Layout::c_contiguous([4]).expect("rank-1 layout");
    let prepared = ops
        .prepare_scalar_into::<AddOp, 1>(
            device,
            StridedView::new(&a, &layout),
            1.0,
            StridedView::new(&out, &layout),
        )
        .expect("prepare scalar");

    ops.dispatch_scalar::<1>(device, &prepared)
        .expect("dispatch");
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(got, [2.0, 3.0, 4.0, 5.0], "{name}: prepared scalar add");

    device
        .write_buffer(&a, &[10.0f32, 20.0, 30.0, 40.0])
        .expect("input rewrite");
    ops.dispatch_scalar::<1>(device, &prepared)
        .expect("rebind dispatch");
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [11.0, 21.0, 31.0, 41.0],
        "{name}: prepared scalar must observe buffer writes while keeping its captured scalar"
    );
}
