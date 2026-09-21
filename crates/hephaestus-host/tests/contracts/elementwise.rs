//! Host instantiation of the shared elementwise conformance clauses
//! (ADR 0061), plus the host-specific dispatch cases its Verification plan
//! names: `f32` `Add` reaching `apply` through the `apply_real` default, and
//! `Pow`/a real-only unary operator on a non-real scalar as the typed host
//! error.

use eunomia::F16;
use hephaestus_conformance::{assert_elementwise_contract, assert_typed_elementwise_contract};
use hephaestus_core::{AddOp, ComputeDevice, ElementwiseOps, ExpOp, PowOp, StridedView};
use hephaestus_host::{HostDevice, HostElementwiseOps};
use leto::Layout;

#[test]
fn host_satisfies_the_elementwise_contract() {
    assert_elementwise_contract(&HostDevice::new(), &HostElementwiseOps);
}

#[test]
fn host_satisfies_the_typed_elementwise_contract() {
    assert_typed_elementwise_contract(&HostDevice::new(), &HostElementwiseOps);
}

/// ADR 0061 Verification plan: `f32` `Add` reaches `apply` through the
/// `apply_real` default (`AddOp` implements only `apply`, never
/// `apply_real`), because the host's per-scalar dispatch routes `f32`
/// through `BinaryExpr::real_value`.
#[test]
fn f32_add_reaches_apply_through_the_apply_real_default() {
    let device = HostDevice::new();
    let ops = HostElementwiseOps;
    let a = device.upload(&[1.5f32, -2.0]).expect("lhs upload");
    let b = device.upload(&[0.5f32, 4.0]).expect("rhs upload");
    let out = device.alloc_zeroed::<f32>(2).expect("output alloc");
    let layout = Layout::c_contiguous([2]).expect("layout");
    ops.binary_into::<AddOp, 1>(
        &device,
        StridedView::new(&a, &layout),
        StridedView::new(&b, &layout),
        StridedView::new(&out, &layout),
    )
    .expect("add dispatch");
    let mut got = [0.0f32; 2];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [2.0, 2.0],
        "f32 Add must compute through apply_real's default"
    );
}

/// `Pow` is real-only (ADR 0061 Decision 3): `apply` reports no definition
/// over the full `NumericElement` set, so `i32` and `F16` dispatch through
/// `BinaryExpr::value` is the typed `DispatchFailed`, never a silent
/// identity or zero (ADR 0061 Verification plan).
#[test]
fn pow_on_a_non_real_scalar_is_the_typed_dispatch_failure() {
    let device = HostDevice::new();
    let ops = HostElementwiseOps;
    let layout = Layout::c_contiguous([1]).expect("layout");

    let a = device.upload(&[2i32]).expect("lhs upload");
    let b = device.upload(&[3i32]).expect("rhs upload");
    let out = device.alloc_zeroed::<i32>(1).expect("output alloc");
    let error = ops
        .binary_into::<PowOp, 1>(
            &device,
            StridedView::new(&a, &layout),
            StridedView::new(&b, &layout),
            StridedView::new(&out, &layout),
        )
        .expect_err("i32 Pow must have no host application");
    let rendered = format!("{error}");
    assert!(
        rendered.contains("PowOp") && rendered.contains("no host value function"),
        "error must name the unsupported operator, got {rendered:?}"
    );

    let a16 = device.upload(&[F16::from_f32(2.0)]).expect("lhs upload");
    let b16 = device.upload(&[F16::from_f32(3.0)]).expect("rhs upload");
    let out16 = device.alloc_zeroed::<F16>(1).expect("output alloc");
    let error16 = ops
        .binary_into::<PowOp, 1>(
            &device,
            StridedView::new(&a16, &layout),
            StridedView::new(&b16, &layout),
            StridedView::new(&out16, &layout),
        )
        .expect_err("F16 Pow must have no host application");
    assert!(format!("{error16}").contains("PowOp"));
}

/// Every unary operator is real-only (ADR 0061 Decision 2): an integer
/// scalar's per-type dispatch reports no application at all, the typed
/// `DispatchFailed` naming the operator.
#[test]
fn a_real_only_unary_operator_on_an_integer_scalar_is_the_typed_dispatch_failure() {
    let device = HostDevice::new();
    let ops = HostElementwiseOps;
    let a = device.upload(&[4i32]).expect("input upload");
    let out = device.alloc_zeroed::<i32>(1).expect("output alloc");
    let layout = Layout::c_contiguous([1]).expect("layout");
    let error = ops
        .unary_into::<ExpOp, 1>(
            &device,
            StridedView::new(&a, &layout),
            StridedView::new(&out, &layout),
        )
        .expect_err("i32 has no real-valued unary application");
    assert!(format!("{error}").contains("ExpOp"));
}
