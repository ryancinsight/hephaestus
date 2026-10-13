use hephaestus_core::test_support::assert_rejects;
use hephaestus_core::{
    ComputeDevice, DialectScalar, ElementwiseOps, EqOp, StridedView, TypedBinaryExpr,
};
use leto::Layout;

/// Operand shape disagreement is a typed rejection and the output is left
/// untouched, not truncated or partially written.
pub(crate) fn shape_mismatch_is_rejected_before_mutation<D, E>(device: &D, ops: &E)
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
    assert_rejects(
        &result,
        "kernel dispatch failed: layout rejected: Incompatible broadcast: from [2] to [3]",
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
pub(crate) fn strided_operands_are_read_through_their_layouts<D, E>(device: &D, ops: &E)
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
