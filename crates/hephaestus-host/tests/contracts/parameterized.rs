//! Host instantiation of the shared runtime-parameter unary conformance
//! clause (ADR 0061).

use hephaestus_conformance::assert_parameterized_unary_contract;
use hephaestus_host::{HostDevice, HostParameterizedUnaryOps};

#[test]
fn host_satisfies_the_parameterized_unary_contract() {
    assert_parameterized_unary_contract(&HostDevice::new(), &HostParameterizedUnaryOps);
}
