//! Host instantiation of the shared retained-reduction conformance clause.

use hephaestus_conformance::assert_retained_reduction_contract;
use hephaestus_host::{HostDenseVectorOps, HostDevice};

#[test]
fn host_satisfies_the_retained_reduction_contract() {
    assert_retained_reduction_contract::<_, _, f32>(&HostDevice::new(), &HostDenseVectorOps);
}

#[test]
fn host_satisfies_the_retained_reduction_contract_f64() {
    assert_retained_reduction_contract::<_, _, f64>(&HostDevice::new(), &HostDenseVectorOps);
}
