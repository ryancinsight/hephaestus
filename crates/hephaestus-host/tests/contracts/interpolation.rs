//! Host instantiation of the shared interpolation conformance clause.

use hephaestus_conformance::assert_interpolation_contract;
use hephaestus_host::{HostDevice, HostInterpolationOps};

#[test]
fn host_satisfies_the_interpolation_contract() {
    assert_interpolation_contract(&HostDevice::new(), &HostInterpolationOps);
}
