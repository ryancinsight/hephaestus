use hephaestus_core::{
    CombineExpr, ComputeDevice, IdentityToken, MaxOp, MinOp, OpIdentity, ProdOp, StridedView, SumOp,
};
use leto::Layout;

use super::backend::ReductionBackend;
use super::fixture::fixture;

/// Reducing along either axis matches the host product and keeps the reduced
/// axis at length one.
pub(crate) fn prod_axis_reduces_both_axes<D, R>(device: &D, ops: &R)
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
pub(crate) fn prod_axis_follows_strided_views<D, R>(device: &D, ops: &R)
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
    // The 4x3 transpose: same bytes, swapped strides.
    let transposed = Layout::try_new([4, 3], [1, 4], 0).expect("valid conformance fixture layout");

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
    device
        .download(&out, &mut got)
        .expect("transposed download");
    assert_eq!(
        got,
        [45.0, 120.0, 231.0, 384.0],
        "{}: transposed reduction must equal the axis-0 result",
        device.backend_name()
    );
}

/// An axis outside the operand rank is a typed rejection, not a clamped
/// dispatch.
pub(crate) fn prod_axis_rejects_out_of_range_axis<D, R>(device: &D, ops: &R)
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
