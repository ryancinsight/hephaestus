//! Host instantiation of the shared adaptive-pooling conformance clause.

use hephaestus_conformance::assert_adaptive_pooling_contract;
use hephaestus_host::{HostAdaptivePoolingOps, HostDevice};

#[test]
fn host_satisfies_the_adaptive_pooling_contract() {
    assert_adaptive_pooling_contract(&HostDevice::new(), &HostAdaptivePoolingOps);
}
