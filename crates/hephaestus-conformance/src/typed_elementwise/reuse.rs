use hephaestus_core::{
    ComputeDevice, DialectScalar, ElementwiseOps, EqOp, StridedView, TypedBinaryExpr,
};
use leto::Layout;

/// The `_into` form writes caller-owned storage and repeated dispatch over
/// unchanged operands is idempotent, not accumulating.
pub(crate) fn into_form_is_idempotent_over_unchanged_operands<D, E>(device: &D, ops: &E)
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
pub(crate) fn prepared_dispatch_is_reusable_and_observes_writes<D, E>(device: &D, ops: &E)
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
