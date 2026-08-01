//! Contract clauses for device-neutral axis prefix/suffix scans.
//!
//! Each clause is generic over the device and the seam, so every backend runs
//! the same assertions. The fixture is the 3x4 matrix with identical
//! `[1,2,3,2]` rows: every running sum stays below 24 and every running
//! product below `12 < 2^24`, so all partials are exact in `f32`, scan order
//! is fully determined by the direction, and every oracle is an exact
//! equality.

use hephaestus_core::{
    CombineExpr, ComputeDevice, CumProdOp, CumSumOp, IdentityToken, OpIdentity, ScanDirection,
    ScanOps, StridedView,
};
use leto::Layout;

/// A backend able to run every clause in this module.
///
/// Bundles the combining operators the dialect must define and the `f32`
/// identities each carries, so a backend satisfies one bound instead of the
/// operator set at every clause.
pub trait ScanBackend<D: ComputeDevice>: ScanOps<D, f32>
where
    CumSumOp: CombineExpr<Self::Dialect>,
    CumProdOp: CombineExpr<Self::Dialect>,
    f32: OpIdentity<CumSumOp>
        + IdentityToken<CumSumOp, Self::Dialect>
        + OpIdentity<CumProdOp>
        + IdentityToken<CumProdOp, Self::Dialect>,
{
}

impl<D, S> ScanBackend<D> for S
where
    D: ComputeDevice,
    S: ScanOps<D, f32>,
    CumSumOp: CombineExpr<S::Dialect>,
    CumProdOp: CombineExpr<S::Dialect>,
    f32: OpIdentity<CumSumOp>
        + IdentityToken<CumSumOp, S::Dialect>
        + OpIdentity<CumProdOp>
        + IdentityToken<CumProdOp, S::Dialect>,
{
}

/// The 3x4 fixture, row-major: three identical `[1,2,3,2]` rows.
fn fixture() -> Vec<f32> {
    [1.0f32, 2.0, 3.0, 2.0].repeat(3)
}

/// Scan the fixture along `axis` in `direction` under `Op` and return the
/// dense 3x4 result.
fn scan<D, S, Op>(device: &D, ops: &S, axis: usize, direction: ScanDirection) -> [f32; 12]
where
    D: ComputeDevice,
    S: ScanOps<D, f32>,
    Op: CombineExpr<S::Dialect>,
    f32: OpIdentity<Op> + IdentityToken<Op, S::Dialect>,
{
    let source = device.upload(&fixture()).expect("fixture upload");
    let out = device.alloc_zeroed::<f32>(12).expect("output alloc");
    let dense = Layout::c_contiguous([3, 4]).expect("dense layout");
    ops.scan_axis_into::<Op, 2>(
        device,
        StridedView::new(&source, &dense),
        axis,
        direction,
        StridedView::new(&out, &dense),
    )
    .expect("scan dispatch");
    let mut got = [0.0f32; 12];
    device.download(&out, &mut got).expect("download");
    got
}

/// Run every scan clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_scan_contract<D, S>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: ScanBackend<D>,
    CumSumOp: CombineExpr<S::Dialect>,
    CumProdOp: CombineExpr<S::Dialect>,
    f32: OpIdentity<CumSumOp>
        + IdentityToken<CumSumOp, S::Dialect>
        + OpIdentity<CumProdOp>
        + IdentityToken<CumProdOp, S::Dialect>,
{
    let name = device.backend_name();
    scan_prepared_rebinds_bound_operands(device, ops);

    // Forward prefix sum along the rows: each [1,2,3,2] row accumulates to
    // [1,3,6,8].
    assert_eq!(
        scan::<_, _, CumSumOp>(device, ops, 1, ScanDirection::Forward),
        [1.0, 3.0, 6.0, 8.0, 1.0, 3.0, 6.0, 8.0, 1.0, 3.0, 6.0, 8.0],
        "{name}: forward row prefix sum"
    );

    // Reverse suffix sum along the rows: [8,7,5,2] per row.
    assert_eq!(
        scan::<_, _, CumSumOp>(device, ops, 1, ScanDirection::Reverse),
        [8.0, 7.0, 5.0, 2.0, 8.0, 7.0, 5.0, 2.0, 8.0, 7.0, 5.0, 2.0],
        "{name}: reverse row suffix sum"
    );

    // Forward prefix product along the rows: [1,2,6,12] per row, every
    // partial exact.
    assert_eq!(
        scan::<_, _, CumProdOp>(device, ops, 1, ScanDirection::Forward),
        [
            1.0, 2.0, 6.0, 12.0, 1.0, 2.0, 6.0, 12.0, 1.0, 2.0, 6.0, 12.0
        ],
        "{name}: forward row prefix product"
    );

    // Forward prefix sum down the columns: column j holds [v, 2v, 3v] for its
    // value v, distinguishing the axis argument from the row case.
    assert_eq!(
        scan::<_, _, CumSumOp>(device, ops, 0, ScanDirection::Forward),
        [1.0, 2.0, 3.0, 2.0, 2.0, 4.0, 6.0, 4.0, 3.0, 6.0, 9.0, 6.0],
        "{name}: forward column prefix sum"
    );

    // An out-of-range axis is rejected and the output untouched.
    let source = device.upload(&fixture()).expect("fixture upload");
    let out = device.upload(&[9.0f32; 12]).expect("output upload");
    let dense = Layout::c_contiguous([3, 4]).expect("dense layout");
    let result = ops.scan_axis_into::<CumSumOp, 2>(
        device,
        StridedView::new(&source, &dense),
        2,
        ScanDirection::Forward,
        StridedView::new(&out, &dense),
    );
    assert!(
        result.is_err(),
        "{name}: axis 2 on a rank-2 operand must be rejected"
    );
    let mut got = [0.0f32; 12];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got, [9.0; 12],
        "{name}: a rejected scan must not touch the output"
    );
}

/// A prepared scan re-dispatches over its bound operands: dispatch is
/// idempotent over unchanged inputs and observes writes made after
/// preparation (the rebind contract shared by every prepared seam form).
fn scan_prepared_rebinds_bound_operands<D, S>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: ScanOps<D, f32>,
    CumSumOp: CombineExpr<S::Dialect>,
    f32: OpIdentity<CumSumOp> + IdentityToken<CumSumOp, S::Dialect>,
{
    let name = device.backend_name();
    let source = device.upload(&fixture()).expect("fixture upload");
    let out = device.alloc_zeroed::<f32>(12).expect("output alloc");
    let dense = Layout::c_contiguous([3, 4]).expect("dense layout");
    let prepared = ops
        .prepare_scan_axis::<CumSumOp, 2>(
            device,
            StridedView::new(&source, &dense),
            1,
            ScanDirection::Forward,
            StridedView::new(&out, &dense),
        )
        .expect("prepare scan");

    let expected = [
        1.0f32, 3.0, 6.0, 8.0, 1.0, 3.0, 6.0, 8.0, 1.0, 3.0, 6.0, 8.0,
    ];
    ops.dispatch_scan::<2>(device, &prepared).expect("dispatch");
    let mut got = [0.0f32; 12];
    device.download(&out, &mut got).expect("download");
    assert_eq!(got, expected, "{name}: prepared forward row prefix sum");

    ops.dispatch_scan::<2>(device, &prepared)
        .expect("re-dispatch");
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: prepared scan must be idempotent over unchanged operands"
    );

    device
        .write_buffer(&source, &[1.0f32; 12])
        .expect("input rewrite");
    ops.dispatch_scan::<2>(device, &prepared)
        .expect("rebind dispatch");
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [
            1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0
        ],
        "{name}: prepared scan must observe writes to its bound operands"
    );
}
