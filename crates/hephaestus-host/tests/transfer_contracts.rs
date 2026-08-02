//! Host instantiation of the shared transfer conformance clauses.

use hephaestus_conformance::assert_transfer_contract;
use hephaestus_host::HostDevice;

#[test]
fn host_satisfies_the_transfer_contract() {
    assert_transfer_contract(&HostDevice::new());
}
