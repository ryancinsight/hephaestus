//! Host instantiation of the shared CTC clauses.

use hephaestus_conformance::assert_ctc_contract;
use hephaestus_host::{HostCtcOps, HostDevice};

#[test]
fn host_satisfies_the_ctc_contract() {
    assert_ctc_contract(&HostDevice::new(), &HostCtcOps);
}
