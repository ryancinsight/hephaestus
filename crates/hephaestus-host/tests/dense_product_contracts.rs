//! Host instantiation of the shared dense-product conformance clauses.
//!
//! Leto joins the kernel-product family's role trait per ADR 0046 §5: the
//! same clause suite the GPU backends run executes against the CPU
//! reference pair.

use hephaestus_conformance::assert_dense_product_contract;
use hephaestus_core::{ComputeDevice, DenseProductOps, StridedView};
use hephaestus_host::{HostDenseProductOps, HostDevice};
use leto::Layout;

#[test]
fn host_satisfies_the_dense_product_contract() {
    assert_dense_product_contract(&HostDevice::new(), &HostDenseProductOps);
}

/// `[[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]`, exact in `f64` — the
/// bound admits any leto `Scalar`, not only the clause's `f32` instantiation.
#[test]
fn matmul_is_exact_at_f64() {
    let device = HostDevice::new();
    let ops = HostDenseProductOps;
    let lhs = device.upload(&[1.0f64, 2.0, 3.0, 4.0]).expect("lhs upload");
    let rhs = device.upload(&[5.0f64, 6.0, 7.0, 8.0]).expect("rhs upload");
    let out = device.alloc_zeroed::<f64>(4).expect("output alloc");
    let two = Layout::c_contiguous([2, 2]).expect("2x2 layout");
    ops.matmul_into(
        &device,
        StridedView::new(&lhs, &two),
        StridedView::new(&rhs, &two),
        StridedView::new(&out, &two),
    )
    .expect("matmul dispatch");
    let mut got = [0.0f64; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(got, [19.0, 22.0, 43.0, 50.0]);
}

/// An output view shaped for a different product is rejected before any
/// element is written; the sentinel values must survive untouched.
#[test]
fn matmul_rejects_a_wrong_shaped_output_without_mutating_it() {
    let device = HostDevice::new();
    let ops = HostDenseProductOps;
    let lhs = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).expect("lhs upload");
    let rhs = device.upload(&[5.0f32, 6.0, 7.0, 8.0]).expect("rhs upload");
    // The true product is 2x2; this output view is 2x3, wrong on either axis
    // pairing, so it cannot be a coincidental match.
    let out = device
        .upload(&[9.0f32, 9.0, 9.0, 9.0, 9.0, 9.0])
        .expect("sentinel");
    let two = Layout::c_contiguous([2, 2]).expect("2x2 layout");
    let wrong_shape = Layout::c_contiguous([2, 3]).expect("2x3 layout");
    let error = ops
        .matmul_into(
            &device,
            StridedView::new(&lhs, &two),
            StridedView::new(&rhs, &two),
            StridedView::new(&out, &wrong_shape),
        )
        .expect_err("wrong-shaped output must be rejected");
    assert!(matches!(
        error,
        hephaestus_core::HephaestusError::DispatchFailed { .. }
    ));
    let mut got = [0.0f32; 6];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got, [9.0; 6],
        "rejected matmul must not mutate the output buffer"
    );
}

/// `A · A` with both operands naming one buffer: squaring is legitimate use,
/// and the operands must be read under one guard rather than two.
#[test]
fn matmul_squares_a_matrix_passed_as_both_operands() {
    let device = HostDevice::new();
    let ops = HostDenseProductOps;
    let a = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).expect("upload");
    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    let two = Layout::c_contiguous([2, 2]).expect("2x2 layout");
    ops.matmul_into(
        &device,
        StridedView::new(&a, &two),
        StridedView::new(&a, &two),
        StridedView::new(&out, &two),
    )
    .expect("A · A dispatch");
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(got, [7.0, 10.0, 15.0, 22.0]);
}

/// A view whose layout addresses more elements than its buffer holds is
/// caller input the device must reject with the typed error, not a panic.
#[test]
fn matmul_rejects_a_view_larger_than_its_buffer() {
    let device = HostDevice::new();
    let ops = HostDenseProductOps;
    let short = device.upload(&[1.0f32, 2.0, 3.0]).expect("three elements");
    let rhs = device.upload(&[5.0f32, 6.0, 7.0, 8.0]).expect("rhs upload");
    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    let two = Layout::c_contiguous([2, 2]).expect("2x2 layout");
    let error = ops
        .matmul_into(
            &device,
            StridedView::new(&short, &two),
            StridedView::new(&rhs, &two),
            StridedView::new(&out, &two),
        )
        .expect_err("a 2x2 view over three elements must be rejected");
    assert!(matches!(
        error,
        hephaestus_core::HephaestusError::DispatchFailed { .. }
    ));
}
