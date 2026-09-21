//! Host instantiation of the shared stateful-update conformance clause (ADR
//! 0061).

use hephaestus_conformance::assert_stateful_update_contract;
use hephaestus_host::{HostDevice, HostStatefulUpdateOps};

#[test]
fn host_satisfies_the_stateful_update_contract() {
    assert_stateful_update_contract(&HostDevice::new(), &HostStatefulUpdateOps);
}
