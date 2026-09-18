//! Host instantiation of the shared seeded-random-initialization clause.

use hephaestus_conformance::assert_random_init_contract;
use hephaestus_host::{HostDevice, HostRandomOps};

#[test]
fn host_satisfies_the_random_init_contract() {
    assert_random_init_contract(&HostDevice::new(), &HostRandomOps);
}
