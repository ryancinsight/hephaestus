use hephaestus_core::{BinaryExpr, ComputeDevice, DialectScalar, ElementwiseOps, StridedView};
use leto::Layout;

/// Dispatch one rank-1 binary arithmetic op through the seam and return the
/// downloaded results.
pub(crate) fn arithmetic<D, E, Op, const LEN: usize>(
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
