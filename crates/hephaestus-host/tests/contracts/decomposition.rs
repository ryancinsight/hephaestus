//! Host instantiation of the shared decomposition conformance clauses.
//!
//! Leto joins the role trait per ADR 0039 section 3: the same clause
//! suite the GPU backends run executes against the CPU reference pair.

use hephaestus_conformance::assert_decomposition_contract;
use hephaestus_core::{ComputeDevice, DecompositionOps, HephaestusError, StridedView};
use hephaestus_host::{HostDecompositionOps, HostDevice};
use leto::Layout;

#[test]
fn host_satisfies_the_decomposition_contract() {
    assert_decomposition_contract(&HostDevice::new(), &HostDecompositionOps);
}

/// A 3x3 layout over a four-element buffer is rejected as a typed error
/// before any element is read; an unchecked view indexed past the buffer.
#[test]
fn a_layout_larger_than_its_buffer_is_rejected() {
    let device = HostDevice::new();
    let short = device.upload(&[1.0f32, 0.0, 0.0, 1.0]).expect("upload");
    let layout = Layout::c_contiguous([3, 3]).expect("3x3 layout");
    let result = HostDecompositionOps.lu(&device, StridedView::new(&short, &layout));
    assert!(
        matches!(result, Err(HephaestusError::DispatchFailed { .. })),
        "a 3x3 view of 4 cells must be a typed rejection"
    );
}
