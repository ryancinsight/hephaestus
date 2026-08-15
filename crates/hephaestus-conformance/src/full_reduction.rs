//! Contract clauses for device-neutral whole-operand reductions.
//!
//! Each clause is generic over the device and the seam, so every backend runs
//! the same assertions. The fixture is the 3x4 matrix
//! `[[1,2,3,2],[1,2,3,2],[1,2,3,2]]`: its sum (24) is exactly representable,
//! its full product is `(1*2*3*2)^3 = 1728 < 2^24` so every partial product is
//! exact in `f32` and reduction order cannot change the result, and its
//! minimum (1) and maximum (3) are unambiguous. No tolerance is applicable.

use hephaestus_core::{
    CombineExpr, ComputeDevice, FullReductionOps, IdentityToken, MaxOp, MinOp, OpIdentity, ProdOp,
    StridedView, SumOp,
};
use leto::Layout;

/// A backend able to run every clause in this module.
///
/// Bundles the combining operators the dialect must define and the `f32`
/// identities each carries, so a backend satisfies one bound instead of the
/// full operator set at every clause.
pub trait FullReductionBackend<D: ComputeDevice>: FullReductionOps<D, f32>
where
    SumOp: CombineExpr<Self::Dialect>,
    ProdOp: CombineExpr<Self::Dialect>,
    MinOp: CombineExpr<Self::Dialect>,
    MaxOp: CombineExpr<Self::Dialect>,
    f32: OpIdentity<SumOp>
        + IdentityToken<SumOp, Self::Dialect>
        + OpIdentity<ProdOp>
        + IdentityToken<ProdOp, Self::Dialect>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, Self::Dialect>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, Self::Dialect>,
{
}

impl<D, R> FullReductionBackend<D> for R
where
    D: ComputeDevice,
    R: FullReductionOps<D, f32>,
    SumOp: CombineExpr<R::Dialect>,
    ProdOp: CombineExpr<R::Dialect>,
    MinOp: CombineExpr<R::Dialect>,
    MaxOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<SumOp>
        + IdentityToken<SumOp, R::Dialect>
        + OpIdentity<ProdOp>
        + IdentityToken<ProdOp, R::Dialect>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, R::Dialect>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, R::Dialect>,
{
}

/// The 3x4 fixture, row-major: three identical `[1,2,3,2]` rows.
fn fixture() -> Vec<f32> {
    [1.0f32, 2.0, 3.0, 2.0].repeat(3)
}

/// Reduce the fixture (viewed through `layout`) under `Op` and return the
/// scalar.
fn reduce<D, R, Op>(device: &D, ops: &R, source: &D::Buffer<f32>, layout: &Layout<2>) -> f32
where
    D: ComputeDevice,
    R: FullReductionOps<D, f32>,
    Op: CombineExpr<R::Dialect>,
    f32: OpIdentity<Op> + IdentityToken<Op, R::Dialect>,
{
    let out = device.alloc_zeroed::<f32>(1).expect("output alloc");
    let out_layout = Layout::c_contiguous([1]).expect("scalar layout");
    ops.reduce_full_into::<Op, 2>(
        device,
        StridedView::new(source, layout),
        StridedView::new(&out, &out_layout),
    )
    .expect("full reduction dispatch");
    let mut got = [0.0f32; 1];
    device.download(&out, &mut got).expect("download");
    got[0]
}

/// Run every full-reduction clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_full_reduction_contract<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: FullReductionBackend<D>,
    SumOp: CombineExpr<R::Dialect>,
    ProdOp: CombineExpr<R::Dialect>,
    MinOp: CombineExpr<R::Dialect>,
    MaxOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<SumOp>
        + IdentityToken<SumOp, R::Dialect>
        + OpIdentity<ProdOp>
        + IdentityToken<ProdOp, R::Dialect>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, R::Dialect>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, R::Dialect>,
{
    let name = device.backend_name();
    let source = device.upload(&fixture()).expect("fixture upload");
    let dense = Layout::c_contiguous([3, 4]).expect("dense layout");

    assert_eq!(
        reduce::<_, _, SumOp>(device, ops, &source, &dense),
        24.0,
        "{name}: full sum"
    );
    assert_eq!(
        reduce::<_, _, ProdOp>(device, ops, &source, &dense),
        1728.0,
        "{name}: full product"
    );
    assert_eq!(
        reduce::<_, _, MinOp>(device, ops, &source, &dense),
        1.0,
        "{name}: full minimum"
    );
    assert_eq!(
        reduce::<_, _, MaxOp>(device, ops, &source, &dense),
        3.0,
        "{name}: full maximum"
    );

    // A strided (transposed) view reduces the same multiset: same bytes,
    // different traversal, identical exact result.
    let transposed = Layout::try_new([4, 3], [1, 4], 0).expect("valid conformance fixture layout");
    assert_eq!(
        reduce::<_, _, SumOp>(device, ops, &source, &transposed),
        24.0,
        "{name}: transposed full sum"
    );
    assert_eq!(
        reduce::<_, _, ProdOp>(device, ops, &source, &transposed),
        1728.0,
        "{name}: transposed full product"
    );

    // A diagonal-strided rank-1 view sums to the trace: uploading the 3x3
    // matrix [[1..3],[4..6],[7..9]] and reducing the stride-4 diagonal view
    // yields 1 + 5 + 9 = 15 exactly. This is the kernel-level path the
    // backend `trace` conveniences take (square check and diagonal-view
    // construction are host-side arithmetic above the same reduction).
    let square = device
        .upload(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        .expect("square upload");
    let diagonal = Layout::try_new([3], [4], 0).expect("valid conformance fixture layout");
    let trace_out = device.alloc_zeroed::<f32>(1).expect("trace output");
    let scalar_layout = Layout::c_contiguous([1]).expect("scalar layout");
    ops.reduce_full_into::<SumOp, 1>(
        device,
        StridedView::new(&square, &diagonal),
        StridedView::new(&trace_out, &scalar_layout),
    )
    .expect("diagonal reduction");
    let mut got_trace = [0.0f32; 1];
    device
        .download(&trace_out, &mut got_trace)
        .expect("trace download");
    assert_eq!(
        got_trace[0], 15.0,
        "{name}: diagonal-strided view must sum to the trace"
    );

    // A prepared reduction may be re-dispatched, and holds its operand
    // bindings rather than a snapshot: after the input is rewritten to all
    // twos, the product becomes 2^12 = 4096, still exact.
    let out1 = device.alloc_zeroed::<f32>(1).expect("output alloc");
    let out_layout = Layout::c_contiguous([1]).expect("scalar layout");
    let prepared = ops
        .prepare_reduce_full::<ProdOp, 2>(
            device,
            StridedView::new(&source, &dense),
            StridedView::new(&out1, &out_layout),
        )
        .expect("prepare full product");
    ops.dispatch_full::<2>(device, &prepared)
        .expect("first dispatch");
    let mut got1 = [0.0f32; 1];
    device.download(&out1, &mut got1).expect("download");
    assert_eq!(got1, [1728.0], "{name}: prepared full product");
    ops.dispatch_full::<2>(device, &prepared)
        .expect("second dispatch");
    device.download(&out1, &mut got1).expect("download");
    assert_eq!(
        got1,
        [1728.0],
        "{name}: prepared full reduction must be idempotent over unchanged input"
    );
    device
        .write_buffer(&source, &[2.0f32; 12])
        .expect("input rewrite");
    ops.dispatch_full::<2>(device, &prepared)
        .expect("third dispatch");
    device.download(&out1, &mut got1).expect("download");
    assert_eq!(
        got1,
        [4096.0],
        "{name}: prepared full reduction must observe writes to its bound input"
    );
    device
        .write_buffer(&source, &fixture())
        .expect("input restore");

    // An output that is not exactly one element is rejected and untouched.
    let out = device.upload(&[9.0f32, 9.0]).expect("output upload");
    let out_layout = Layout::c_contiguous([2]).expect("output layout");
    let result = ops.reduce_full_into::<SumOp, 2>(
        device,
        StridedView::new(&source, &dense),
        StridedView::new(&out, &out_layout),
    );
    assert!(
        result.is_err(),
        "{name}: a multi-element output must be rejected"
    );
    let mut got = [0.0f32; 2];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [9.0, 9.0],
        "{name}: a rejected full reduction must not touch the output"
    );
}
