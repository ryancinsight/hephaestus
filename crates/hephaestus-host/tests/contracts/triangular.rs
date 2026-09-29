//! Host instantiation of the shared triangular-masking conformance clause.

use hephaestus_conformance::assert_triangular_contract;
use hephaestus_host::{HostDevice, HostTriangularOps};

#[test]
fn host_satisfies_the_triangular_contract() {
    assert_triangular_contract(&HostDevice::new(), &HostTriangularOps);
}
