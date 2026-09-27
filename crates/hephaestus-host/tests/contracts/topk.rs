//! Host instantiation of the shared top-k conformance clause.

use hephaestus_conformance::assert_topk_contract;
use hephaestus_host::{HostDevice, HostTopKOps};

#[test]
fn host_satisfies_the_topk_contract() {
    assert_topk_contract(&HostDevice::new(), &HostTopKOps);
}
