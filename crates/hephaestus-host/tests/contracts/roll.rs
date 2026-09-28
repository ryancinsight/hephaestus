//! Host instantiation of the shared roll conformance clause.

use hephaestus_conformance::assert_roll_contract;
use hephaestus_host::{HostDevice, HostRollOps};

#[test]
fn host_satisfies_the_roll_contract() {
    assert_roll_contract(&HostDevice::new(), &HostRollOps);
}
