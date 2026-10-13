use hephaestus_core::{
    CombineExpr, ComputeDevice, IdentityToken, MaxOp, MinOp, OpIdentity, ProdOp, StridedView, SumOp,
};
use leto::Layout;

use super::backend::ReductionBackend;
use super::fixture::fixture;

/// A prepared reduction may be dispatched repeatedly, and holds its operand
/// allocations rather than a snapshot of their contents.
pub(crate) fn prepared_reduction_is_reusable_and_observes_writes<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: ReductionBackend<D>,
    ProdOp: CombineExpr<R::Dialect>,
    SumOp: CombineExpr<R::Dialect>,
    MinOp: CombineExpr<R::Dialect>,
    MaxOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, R::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, R::Dialect>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, R::Dialect>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, R::Dialect>,
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
pub(crate) fn prepared_reduction_is_parameterized_by_operator<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: ReductionBackend<D>,
    ProdOp: CombineExpr<R::Dialect>,
    SumOp: CombineExpr<R::Dialect>,
    MinOp: CombineExpr<R::Dialect>,
    MaxOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, R::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, R::Dialect>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, R::Dialect>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, R::Dialect>,
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
    device
        .download(&sum_out, &mut got_sum)
        .expect("sum download");
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
    ops.dispatch_prepared(device, &prod)
        .expect("product dispatch");
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
pub(crate) fn prepared_reduction_rejects_mismatched_output_shape<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: ReductionBackend<D>,
    ProdOp: CombineExpr<R::Dialect>,
    SumOp: CombineExpr<R::Dialect>,
    MinOp: CombineExpr<R::Dialect>,
    MaxOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, R::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, R::Dialect>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, R::Dialect>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, R::Dialect>,
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
