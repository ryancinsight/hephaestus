use eunomia::Pod;
use hephaestus_core::{ComputeDevice, DialectScalar, ElementwiseOps, StridedView, TypedBinaryExpr};
use leto::Layout;

/// Dispatch one rank-1 typed comparison through the seam and return the
/// downloaded indicators.
pub(crate) fn compare<D, E, T, Op, const LEN: usize>(
    device: &D,
    ops: &E,
    lhs: &[T],
    rhs: &[T],
) -> [T; LEN]
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
