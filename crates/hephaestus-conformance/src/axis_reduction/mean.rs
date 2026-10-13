use hephaestus_core::{
    CombineExpr, ComputeDevice, IdentityToken, MaxOp, MinOp, OpIdentity, ProdOp, StridedView, SumOp,
};
use leto::Layout;

use super::backend::ReductionBackend;
use super::fixture::fixture;

/// Means along both axes are exact: every fixture column sum is divisible
/// by 3 and every row sum by 4 in dyadic arithmetic.
pub(crate) fn mean_axis_divides_by_the_reduced_length<D, R>(device: &D, ops: &R)
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

    // Axis 0: column sums 15, 18, 21, 24 over 3 rows.
    let out = device.alloc_zeroed::<f32>(4).expect("axis-0 output");
    let out_layout = Layout::c_contiguous([1, 4]).expect("axis-0 layout");
    ops.mean_axis_into(
        device,
        StridedView::new(&input, &input_layout),
        0,
        StridedView::new(&out, &out_layout),
    )
    .expect("axis-0 mean");
    let mut got = [0.0f32; 4];
    device
        .download(&out, &mut got)
        .expect("axis-0 mean download");
    assert_eq!(got, [5.0, 6.0, 7.0, 8.0], "{name}: axis-0 mean");

    // Axis 1: row sums 10, 26, 42 over 4 columns.
    let rows = device.alloc_zeroed::<f32>(3).expect("axis-1 output");
    let rows_layout = Layout::c_contiguous([3, 1]).expect("axis-1 layout");
    ops.mean_axis_into(
        device,
        StridedView::new(&input, &input_layout),
        1,
        StridedView::new(&rows, &rows_layout),
    )
    .expect("axis-1 mean");
    let mut got_rows = [0.0f32; 3];
    device
        .download(&rows, &mut got_rows)
        .expect("axis-1 mean download");
    assert_eq!(got_rows, [2.5, 6.5, 10.5], "{name}: axis-1 mean");
}
