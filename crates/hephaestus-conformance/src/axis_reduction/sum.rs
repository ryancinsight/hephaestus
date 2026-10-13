use hephaestus_core::{
    AxisReductionOps, CombineExpr, ComputeDevice, IdentityToken, OpIdentity, StridedView, SumOp,
};
use leto::Layout;

/// Floating-point axis-sum differential against Leto beyond the block width.
///
/// The clauses above use integer-valued fixtures, where every partial sum is
/// exact and therefore order-independent: they pin the *result* but not the
/// summation model, so a backend that summed in any order would pass. This
/// clause uses a genuine `f32` fixture on an axis longer than every backend's
/// default block width (`BlockWidth::DEFAULT` = 256), where the per-lane
/// stride accumulation and the tree reduction reassociate the sum relative to
/// Leto's sequential CPU reference, so the oracle is a derived bound rather
/// than an equality.
///
/// The bound is the naive-summation error model `O(n·ε·Σ|xᵢ|)` with headroom
/// for the extra tree-reduction level — the same derivation the backends'
/// hand-written grid-stride fixtures state — and it holds for *any* summation
/// order, so it is backend-neutral rather than fitted to one kernel.
pub(crate) fn axis_sum_matches_the_leto_reference_beyond_the_block_width<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: AxisReductionOps<D, f32>,
    SumOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<SumOp> + IdentityToken<SumOp, R::Dialect>,
{
    let name = device.backend_name();
    let axis_len = 500usize;
    let rows = 3usize;
    let host: Vec<f32> = (0..rows * axis_len)
        .map(|index| (index % 97) as f32 * 0.1 + 0.01)
        .collect();
    let source = device.upload(&host).expect("fixture upload");
    let out = device.alloc_zeroed::<f32>(rows).expect("output alloc");
    let input = Layout::c_contiguous([rows, axis_len]).expect("input layout");
    let reduced = Layout::c_contiguous([rows, 1]).expect("reduced layout");
    ops.reduce_axis_into::<SumOp>(
        device,
        StridedView::new(&source, &input),
        1,
        StridedView::new(&out, &reduced),
    )
    .expect("axis sum dispatch");

    let leto_input = leto::Array::from_shape_vec([rows, axis_len], host).expect("leto fixture");
    let expected = leto_ops::sum_axis(&leto_input.view(), 1)
        .expect("leto axis sum")
        .into_vec();
    let mut got = vec![0.0f32; rows];
    device.download(&out, &mut got).expect("download");
    for row in 0..rows {
        let view = leto_input.view();
        let abs_sum: f32 = (0..axis_len).map(|col| view[[row, col]].abs()).sum();
        let tolerance = abs_sum * axis_len as f32 * f32::EPSILON * 4.0;
        assert!(
            (got[row] - expected[row]).abs() <= tolerance,
            "{name}: axis sum row {row} = {} must match Leto's sequential sum {} within \
             {tolerance} (derived from O(n·ε·Σ|x|) with tree-reduction headroom)",
            got[row],
            expected[row]
        );
    }
}
