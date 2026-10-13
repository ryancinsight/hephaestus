use hephaestus_core::{
    CombineExpr, ComputeDevice, IdentityToken, MaxOp, MinOp, OpIdentity, ProdOp, StridedView, SumOp,
};
use leto::Layout;

use super::backend::ReductionBackend;
use super::fixture::fixture;

/// Min and max along both axes are exact order statistics of the fixture.
pub(crate) fn min_and_max_axis_reduce_exactly<D, R>(device: &D, ops: &R)
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
    let name = device.backend_name();
    let input = device.upload(&fixture()).expect("fixture upload");
    let input_layout = Layout::c_contiguous([3, 4]).expect("input layout");

    let out = device.alloc_zeroed::<f32>(4).expect("axis-0 output");
    let out_layout = Layout::c_contiguous([1, 4]).expect("axis-0 layout");
    ops.reduce_axis_into::<MinOp>(
        device,
        StridedView::new(&input, &input_layout),
        0,
        StridedView::new(&out, &out_layout),
    )
    .expect("axis-0 min");
    let mut got = [0.0f32; 4];
    device
        .download(&out, &mut got)
        .expect("axis-0 min download");
    assert_eq!(got, [1.0, 2.0, 3.0, 4.0], "{name}: axis-0 min");

    ops.reduce_axis_into::<MaxOp>(
        device,
        StridedView::new(&input, &input_layout),
        0,
        StridedView::new(&out, &out_layout),
    )
    .expect("axis-0 max");
    device
        .download(&out, &mut got)
        .expect("axis-0 max download");
    assert_eq!(got, [9.0, 10.0, 11.0, 12.0], "{name}: axis-0 max");

    let rows = device.alloc_zeroed::<f32>(3).expect("axis-1 output");
    let rows_layout = Layout::c_contiguous([3, 1]).expect("axis-1 layout");
    ops.reduce_axis_into::<MinOp>(
        device,
        StridedView::new(&input, &input_layout),
        1,
        StridedView::new(&rows, &rows_layout),
    )
    .expect("axis-1 min");
    let mut got_rows = [0.0f32; 3];
    device
        .download(&rows, &mut got_rows)
        .expect("axis-1 min download");
    assert_eq!(got_rows, [1.0, 5.0, 9.0], "{name}: axis-1 min");

    ops.reduce_axis_into::<MaxOp>(
        device,
        StridedView::new(&input, &input_layout),
        1,
        StridedView::new(&rows, &rows_layout),
    )
    .expect("axis-1 max");
    device
        .download(&rows, &mut got_rows)
        .expect("axis-1 max download");
    assert_eq!(got_rows, [4.0, 8.0, 12.0], "{name}: axis-1 max");
}
