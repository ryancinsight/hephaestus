//! Host instantiation of the shared cross-product conformance clause.

use hephaestus_conformance::assert_cross_product_contract;
use hephaestus_host::{HostCrossProductOps, HostDevice};

#[test]
fn host_satisfies_the_cross_product_contract() {
    assert_cross_product_contract(&HostDevice::new(), &HostCrossProductOps);
}
