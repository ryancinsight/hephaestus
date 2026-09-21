//! Host instantiation of the shared scan conformance clause, plus the
//! host-specific cases the shared fixture does not reach: a rank other than
//! 2, the `CombineExpr<Host>::value` `None` path (ADR 0061), and integer
//! wraparound.

use hephaestus_conformance::assert_scan_contract;
use hephaestus_core::{
    CombineExpr, ComputeDevice, CumProdOp, CumSumOp, HephaestusError, Host, OpIdentity,
    ScanDirection, ScanOps, StridedView,
};
use hephaestus_host::{HostDevice, HostScanOps};
use leto::Layout;

#[test]
fn host_satisfies_the_scan_contract() {
    assert_scan_contract(&HostDevice::new(), &HostScanOps);
}

/// A combine marker that implements the source trait for `Host` directly,
/// without `CombineValue` (ADR 0061 Consequences: legal, `value` reports
/// `None`).
#[derive(Clone, Copy, Debug)]
struct LocalNoValueOp;

impl CombineExpr<Host> for LocalNoValueOp {
    const EXPR: &'static str = "unused: the host never renders this operator";
}

impl OpIdentity<LocalNoValueOp> for f32 {
    const IDENTITY: Self = 0.0;
}

/// An operator with no host value function is a typed dispatch failure, not
/// a silent identity fallback.
#[test]
fn an_operator_without_a_host_value_function_is_a_typed_dispatch_failure() {
    let device = HostDevice::new();
    let ops = HostScanOps;
    let source = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).expect("upload");
    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    let dense = Layout::c_contiguous([2, 2]).expect("dense layout");

    let error = ops
        .scan_axis_into::<LocalNoValueOp, 2>(
            &device,
            StridedView::new(&source, &dense),
            1,
            ScanDirection::Forward,
            StridedView::new(&out, &dense),
        )
        .expect_err("an operator with no host value function must be rejected");
    let rendered = format!("{error}");
    assert!(
        rendered.contains("LocalNoValueOp") && rendered.contains("no host value function"),
        "error must name the unsupported operator, got {rendered:?}"
    );
}

/// The device-neutral seam is generic over rank, but the shared
/// `plan_axis_scan` validates rank-2 layouts only; a rank-1 operand is a
/// typed rejection, matching `hephaestus-wgpu`'s `WgpuScanOps`.
#[test]
fn a_rank_other_than_two_is_rejected() {
    let device = HostDevice::new();
    let ops = HostScanOps;
    let source = device.upload(&[1.0f32, 2.0, 3.0]).expect("upload");
    let out = device.alloc_zeroed::<f32>(3).expect("output alloc");
    let dense = Layout::c_contiguous([3]).expect("rank-1 layout");

    let result = ops.scan_axis_into::<CumSumOp, 1>(
        &device,
        StridedView::new(&source, &dense),
        0,
        ScanDirection::Forward,
        StridedView::new(&out, &dense),
    );
    assert!(
        matches!(result, Err(HephaestusError::DispatchFailed { .. })),
        "{result:?}"
    );
    let rendered = format!("{}", result.expect_err("checked above"));
    assert!(
        rendered.contains("rank-2 operands"),
        "error must name the rank contract, got {rendered:?}"
    );
}

fn scan_u32(
    device: &HostDevice,
    ops: &HostScanOps,
    values: &[u32; 2],
    direction: ScanDirection,
) -> [u32; 2] {
    let source = device.upload(values.as_slice()).expect("upload");
    let out = device.alloc_zeroed::<u32>(2).expect("output alloc");
    let dense = Layout::c_contiguous([1, 2]).expect("dense layout");
    ops.scan_axis_into::<CumSumOp, 2>(
        device,
        StridedView::new(&source, &dense),
        1,
        direction,
        StridedView::new(&out, &dense),
    )
    .expect("scan dispatch");
    let mut got = [0u32; 2];
    device.download(&out, &mut got).expect("download");
    got
}

/// A cumulative sum wraps in two's complement (ADR 0061 Decision 6), matching
/// WGSL's integer arithmetic rather than panicking on overflow.
#[test]
fn integer_scans_wrap_on_overflow() {
    let device = HostDevice::new();
    let ops = HostScanOps;
    assert_eq!(
        scan_u32(&device, &ops, &[u32::MAX, 1], ScanDirection::Forward),
        [u32::MAX, 0],
        "the running sum must wrap to 0 at the second position, not panic"
    );
}

/// `CumProdOp` also resolves through the host value function; exercised here
/// once as a second operator over the same seam, distinguishing the operator
/// parameter from `CumSumOp`.
#[test]
fn cumulative_product_is_a_real_second_operator() {
    let device = HostDevice::new();
    let ops = HostScanOps;
    let source = device.upload(&[2.0f32, 3.0, 4.0]).expect("upload");
    let out = device.alloc_zeroed::<f32>(3).expect("output alloc");
    let dense = Layout::c_contiguous([1, 3]).expect("dense layout");
    ops.scan_axis_into::<CumProdOp, 2>(
        &device,
        StridedView::new(&source, &dense),
        1,
        ScanDirection::Forward,
        StridedView::new(&out, &dense),
    )
    .expect("scan dispatch");
    let mut got = [0.0f32; 3];
    device.download(&out, &mut got).expect("download");
    assert_eq!(got, [2.0, 6.0, 24.0]);
}
