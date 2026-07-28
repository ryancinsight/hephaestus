//! Contract clauses for device-neutral rank-2 axis reductions.
//!
//! Each clause is generic over the device and the seam, so every backend runs
//! the same assertions. The fixture is the 3x4 matrix `[[1,2,3,4],[5,6,7,8],
//! [9,10,11,12]]`: small enough to state expected values inline, and chosen so
//! every product stays below `2^24` and is therefore exact in `f32`.

use hephaestus_core::{
    AxisReductionOps, CombineExpr, ComputeDevice, IdentityToken, OpIdentity, ProdOp, StridedView,
    SumOp,
};
use leto::Layout;

/// A backend able to run every clause in this module.
///
/// The clauses need four facts that the seam alone does not state: that the
/// backend's dialect defines the sum and product reduction expressions, and that
/// `f32` carries an identity and identity token for each. Restating those at
/// every clause would be generic-parameter chaining; bundling them here keeps
/// each clause's signature to the two parameters it actually varies over, and
/// gives a backend one bound to satisfy.
pub trait ReductionBackend<D: ComputeDevice>: AxisReductionOps<D, f32>
where
    ProdOp: CombineExpr<Self::Dialect>,
    SumOp: CombineExpr<Self::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, Self::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, Self::Dialect>,
{
}

impl<D, R> ReductionBackend<D> for R
where
    D: ComputeDevice,
    R: AxisReductionOps<D, f32>,
    ProdOp: CombineExpr<R::Dialect>,
    SumOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, R::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, R::Dialect>,
{
}

/// The 3x4 fixture, row-major.
fn fixture() -> Vec<f32> {
    (1..=12).map(|value| value as f32).collect()
}

/// Run every axis-reduction clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a device.
pub fn assert_axis_reduction_contract<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: ReductionBackend<D>,
    ProdOp: CombineExpr<R::Dialect>,
    SumOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, R::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, R::Dialect>,
{
    prod_axis_reduces_both_axes(device, ops);
    prod_axis_follows_strided_views(device, ops);
    prod_axis_rejects_out_of_range_axis(device, ops);
    prepared_reduction_is_reusable_and_observes_writes(device, ops);
    prepared_reduction_is_parameterized_by_operator(device, ops);
    prepared_reduction_rejects_mismatched_output_shape(device, ops);
}

/// Reducing along either axis matches the host product and keeps the reduced
/// axis at length one.
fn prod_axis_reduces_both_axes<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: ReductionBackend<D>,
    ProdOp: CombineExpr<R::Dialect>,
    SumOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, R::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, R::Dialect>,
{
    let input = device.upload(&fixture()).expect("fixture upload");
    let input_layout = Layout::c_contiguous([3, 4]).expect("input layout");

    // Axis 0 collapses rows: columnwise products 1*5*9, 2*6*10, 3*7*11, 4*8*12.
    let axis0 = device.alloc_zeroed::<f32>(4).expect("axis-0 output");
    let axis0_layout = Layout::c_contiguous([1, 4]).expect("axis-0 layout");
    ops.prod_axis_into(
        device,
        StridedView::new(&input, &input_layout),
        0,
        StridedView::new(&axis0, &axis0_layout),
    )
    .expect("axis-0 product");
    let mut got = [0.0f32; 4];
    device.download(&axis0, &mut got).expect("axis-0 download");
    assert_eq!(
        got,
        [45.0, 120.0, 231.0, 384.0],
        "{}: axis-0 product",
        device.backend_name()
    );

    // Axis 1 collapses columns: rowwise products.
    let axis1 = device.alloc_zeroed::<f32>(3).expect("axis-1 output");
    let axis1_layout = Layout::c_contiguous([3, 1]).expect("axis-1 layout");
    ops.prod_axis_into(
        device,
        StridedView::new(&input, &input_layout),
        1,
        StridedView::new(&axis1, &axis1_layout),
    )
    .expect("axis-1 product");
    let mut got = [0.0f32; 3];
    device.download(&axis1, &mut got).expect("axis-1 download");
    assert_eq!(
        got,
        [24.0, 1680.0, 11880.0],
        "{}: axis-1 product",
        device.backend_name()
    );
}

/// A reduction reads through the layout, not in buffer order: reducing the
/// transpose along axis 1 reproduces the original's axis-0 result.
fn prod_axis_follows_strided_views<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: ReductionBackend<D>,
    ProdOp: CombineExpr<R::Dialect>,
    SumOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, R::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, R::Dialect>,
{
    let input = device.upload(&fixture()).expect("fixture upload");
    // The 4x3 transpose: same bytes, swapped strides.
    let transposed = Layout::new([4, 3], [1, 4], 0);

    let out = device.alloc_zeroed::<f32>(4).expect("transposed output");
    let out_layout = Layout::c_contiguous([4, 1]).expect("transposed output layout");
    ops.prod_axis_into(
        device,
        StridedView::new(&input, &transposed),
        1,
        StridedView::new(&out, &out_layout),
    )
    .expect("transposed product");

    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("transposed download");
    assert_eq!(
        got,
        [45.0, 120.0, 231.0, 384.0],
        "{}: transposed reduction must equal the axis-0 result",
        device.backend_name()
    );
}

/// An axis outside the operand rank is a typed rejection, not a clamped
/// dispatch.
fn prod_axis_rejects_out_of_range_axis<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: ReductionBackend<D>,
    ProdOp: CombineExpr<R::Dialect>,
    SumOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, R::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, R::Dialect>,
{
    let input = device.upload(&fixture()).expect("fixture upload");
    let input_layout = Layout::c_contiguous([3, 4]).expect("input layout");
    let out = device.alloc_zeroed::<f32>(4).expect("output");
    let out_layout = Layout::c_contiguous([1, 4]).expect("output layout");

    let result = ops.prod_axis_into(
        device,
        StridedView::new(&input, &input_layout),
        2,
        StridedView::new(&out, &out_layout),
    );
    assert!(
        result.is_err(),
        "{}: axis 2 on a rank-2 operand must be rejected",
        device.backend_name()
    );
}

/// A prepared reduction may be dispatched repeatedly, and holds its operand
/// allocations rather than a snapshot of their contents.
fn prepared_reduction_is_reusable_and_observes_writes<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: ReductionBackend<D>,
    ProdOp: CombineExpr<R::Dialect>,
    SumOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, R::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, R::Dialect>,
{
    let input = device.upload(&fixture()).expect("fixture upload");
    let input_layout = Layout::c_contiguous([3, 4]).expect("input layout");
    let out = device.alloc_zeroed::<f32>(4).expect("output");
    let out_layout = Layout::c_contiguous([1, 4]).expect("output layout");

    let prepared = ops
        .prepare_reduce_axis_into::<ProdOp>(
            device,
            StridedView::new(&input, &input_layout),
            0,
            StridedView::new(&out, &out_layout),
        )
        .expect("prepare product reduction");

    ops.dispatch_prepared(device, &prepared)
        .expect("first dispatch");
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("first download");
    assert_eq!(
        got,
        [45.0, 120.0, 231.0, 384.0],
        "{}: prepared reduction first dispatch",
        device.backend_name()
    );

    // Re-dispatch is stable, not accumulating.
    ops.dispatch_prepared(device, &prepared)
        .expect("second dispatch");
    device.download(&out, &mut got).expect("second download");
    assert_eq!(
        got,
        [45.0, 120.0, 231.0, 384.0],
        "{}: prepared reduction must be idempotent over unchanged input",
        device.backend_name()
    );

    // The plan holds the buffer: a later write is observed.
    device
        .write_buffer(&input, &[2.0f32; 12])
        .expect("input rewrite");
    ops.dispatch_prepared(device, &prepared)
        .expect("third dispatch");
    device.download(&out, &mut got).expect("third download");
    assert_eq!(
        got,
        [8.0, 8.0, 8.0, 8.0],
        "{}: prepared reduction must observe writes to its bound input",
        device.backend_name()
    );
}

/// The combining operator is a real type parameter: `SumOp` and `ProdOp` over
/// identical operands must not agree.
fn prepared_reduction_is_parameterized_by_operator<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: ReductionBackend<D>,
    ProdOp: CombineExpr<R::Dialect>,
    SumOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, R::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, R::Dialect>,
{
    let input = device.upload(&fixture()).expect("fixture upload");
    let input_layout = Layout::c_contiguous([3, 4]).expect("input layout");

    let sum_out = device.alloc_zeroed::<f32>(4).expect("sum output");
    let sum_layout = Layout::c_contiguous([1, 4]).expect("sum layout");
    let sum = ops
        .prepare_reduce_axis_into::<SumOp>(
            device,
            StridedView::new(&input, &input_layout),
            0,
            StridedView::new(&sum_out, &sum_layout),
        )
        .expect("prepare sum reduction");
    ops.dispatch_prepared(device, &sum).expect("sum dispatch");
    let mut got_sum = [0.0f32; 4];
    device.download(&sum_out, &mut got_sum).expect("sum download");
    assert_eq!(
        got_sum,
        [15.0, 18.0, 21.0, 24.0],
        "{}: prepared sum reduction",
        device.backend_name()
    );

    let prod_out = device.alloc_zeroed::<f32>(4).expect("product output");
    let prod_layout = Layout::c_contiguous([1, 4]).expect("product layout");
    let prod = ops
        .prepare_reduce_axis_into::<ProdOp>(
            device,
            StridedView::new(&input, &input_layout),
            0,
            StridedView::new(&prod_out, &prod_layout),
        )
        .expect("prepare product reduction");
    ops.dispatch_prepared(device, &prod).expect("product dispatch");
    let mut got_prod = [0.0f32; 4];
    device
        .download(&prod_out, &mut got_prod)
        .expect("product download");
    assert_eq!(
        got_prod,
        [45.0, 120.0, 231.0, 384.0],
        "{}: prepared product reduction",
        device.backend_name()
    );

    assert_ne!(
        got_sum,
        got_prod,
        "{}: the operator parameter must change the result",
        device.backend_name()
    );
}

/// An output whose shape is not the reduced shape is rejected at plan time.
fn prepared_reduction_rejects_mismatched_output_shape<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: ReductionBackend<D>,
    ProdOp: CombineExpr<R::Dialect>,
    SumOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, R::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, R::Dialect>,
{
    let input = device.upload(&fixture()).expect("fixture upload");
    let input_layout = Layout::c_contiguous([3, 4]).expect("input layout");
    // Reducing a 3x4 along axis 0 must produce 1x4, not 1x3.
    let out = device.alloc_zeroed::<f32>(3).expect("output");
    let wrong = Layout::c_contiguous([1, 3]).expect("wrong output layout");

    let result = ops.prepare_reduce_axis_into::<ProdOp>(
        device,
        StridedView::new(&input, &input_layout),
        0,
        StridedView::new(&out, &wrong),
    );
    assert!(
        result.is_err(),
        "{}: a 1x3 output for a 1x4 reduced shape must be rejected",
        device.backend_name()
    );
}
