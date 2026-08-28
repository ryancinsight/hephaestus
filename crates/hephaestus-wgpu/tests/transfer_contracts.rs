//! WGPU instantiation of the shared transfer conformance clauses.

use hephaestus_conformance::assert_transfer_contract;
pub(super) fn wgpu_satisfies_the_transfer_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_transfer_contract(&device);
}
