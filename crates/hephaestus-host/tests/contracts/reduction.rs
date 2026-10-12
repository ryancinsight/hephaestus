//! Host instantiation of the shared full- and axis-reduction conformance
//! clauses, plus the host-specific cases the shared fixtures do not reach:
//! the `CombineExpr<Host>::value` `None` path (ADR 0061) and integer
//! wraparound.

use hephaestus_conformance::{assert_axis_reduction_contract, assert_full_reduction_contract};
use hephaestus_core::{
    CombineExpr, ComputeDevice, FullReductionOps, HephaestusError, Host, OpIdentity, ProdOp,
    Result, StridedView, SumOp,
};
use hephaestus_host::{HostDevice, HostFullReductionOps};
use leto::Layout;

#[test]
fn host_satisfies_the_full_reduction_contract() {
    assert_full_reduction_contract(&HostDevice::new(), &HostFullReductionOps);
}

#[test]
fn host_satisfies_the_axis_reduction_contract() {
    assert_axis_reduction_contract(&HostDevice::new(), &hephaestus_host::HostAxisReductionOps);
}

/// A combine marker that implements the source trait for `Host` directly,
/// without `CombineValue` — the overlap analysis ADR 0061 Consequences
/// documents: legal, and `value` reports `None`.
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
    let ops = HostFullReductionOps;
    let source = device.upload(&[1.0f32, 2.0, 3.0]).expect("upload");
    let source_layout = Layout::c_contiguous([3]).expect("layout");
    let out = device.alloc_zeroed::<f32>(1).expect("output alloc");
    let out_layout = Layout::c_contiguous([1]).expect("scalar layout");

    let error = ops
        .reduce_full_into::<LocalNoValueOp, 1>(
            &device,
            StridedView::new(&source, &source_layout),
            StridedView::new(&out, &out_layout),
        )
        .expect_err("an operator with no host value function must be rejected");
    let rendered = format!("{error}");
    assert!(
        rendered.contains("LocalNoValueOp") && rendered.contains("no host value function"),
        "error must name the unsupported operator, got {rendered:?}"
    );
    let mut got = [9.0f32; 1];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [0.0],
        "a rejected reduction must not touch the output past its zeroed allocation"
    );
}

fn reduce_u32<Op>(device: &HostDevice, ops: &HostFullReductionOps, values: &[u32; 2]) -> u32
where
    Op: CombineExpr<Host>,
    u32: OpIdentity<Op> + hephaestus_core::IdentityToken<Op, Host>,
{
    let source = device.upload(values.as_slice()).expect("upload");
    let source_layout = Layout::c_contiguous([2]).expect("layout");
    let out = device.alloc_zeroed::<u32>(1).expect("output alloc");
    let out_layout = Layout::c_contiguous([1]).expect("scalar layout");
    ops.reduce_full_into::<Op, 1>(
        device,
        StridedView::new(&source, &source_layout),
        StridedView::new(&out, &out_layout),
    )
    .expect("reduction dispatch");
    let mut got = [0u32; 1];
    device.download(&out, &mut got).expect("download");
    got[0]
}

/// `SumOp` and `ProdOp` wrap in two's complement (ADR 0061 Decision 6, via
/// `eunomia::NumericElement::wrapping_add`/`wrapping_mul`), matching WGSL's
/// integer arithmetic rather than panicking on overflow.
#[test]
fn integer_reductions_wrap_on_overflow() {
    let device = HostDevice::new();
    let ops = HostFullReductionOps;
    assert_eq!(
        reduce_u32::<SumOp>(&device, &ops, &[u32::MAX, 1]),
        0,
        "u32::MAX + 1 must wrap to 0, not panic"
    );
    assert_eq!(
        reduce_u32::<ProdOp>(&device, &ops, &[u32::MAX, 2]),
        u32::MAX.wrapping_mul(2),
        "u32::MAX * 2 must wrap, not panic"
    );
}

/// `mean_axis_into` rejects an empty reduced axis rather than dividing by
/// zero, matching `hephaestus-wgpu`'s `mean_axis_into` free function.
#[test]
fn mean_axis_rejects_an_empty_reduced_axis() {
    use hephaestus_core::AxisReductionOps;
    let device = HostDevice::new();
    let ops = hephaestus_host::HostAxisReductionOps;
    let input = device.upload::<f32>(&[]).expect("upload");
    let input_layout = Layout::c_contiguous([0, 3]).expect("input layout");
    let out = device.alloc_zeroed::<f32>(3).expect("output alloc");
    let out_layout = Layout::c_contiguous([1, 3]).expect("output layout");

    let result: Result<()> = ops.mean_axis_into(
        &device,
        StridedView::new(&input, &input_layout),
        0,
        StridedView::new(&out, &out_layout),
    );
    assert!(
        matches!(result, Err(HephaestusError::DispatchFailed { .. })),
        "{result:?}"
    );
}

/// `mean_axis_into` divides the column sums by the reduced axis length as the
/// element type: exact for floats, truncating for integers.
#[test]
fn mean_axis_divides_by_the_reduced_axis_length() {
    use hephaestus_core::AxisReductionOps;
    let device = HostDevice::new();
    let ops = hephaestus_host::HostAxisReductionOps;
    let input_layout = Layout::c_contiguous([3, 2]).expect("input layout");
    let out_layout = Layout::c_contiguous([1, 2]).expect("output layout");

    let floats = device
        .upload::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("upload");
    let float_out = device.alloc_zeroed::<f32>(2).expect("output alloc");
    ops.mean_axis_into(
        &device,
        StridedView::new(&floats, &input_layout),
        0,
        StridedView::new(&float_out, &out_layout),
    )
    .expect("float mean");
    let mut mean = [0.0_f32; 2];
    device.download(&float_out, &mut mean).expect("download");
    assert_eq!(mean, [3.0, 4.0]);

    let integers = device.upload::<i32>(&[1, 2, 3, 4, 4, 6]).expect("upload");
    let integer_out = device.alloc_zeroed::<i32>(2).expect("output alloc");
    ops.mean_axis_into(
        &device,
        StridedView::new(&integers, &input_layout),
        0,
        StridedView::new(&integer_out, &out_layout),
    )
    .expect("integer mean");
    let mut mean = [0_i32; 2];
    device.download(&integer_out, &mut mean).expect("download");
    // Column sums 8 and 12 over three rows: 8 / 3 truncates to 2, 12 / 3 is 4.
    assert_eq!(mean, [2, 4]);
}
