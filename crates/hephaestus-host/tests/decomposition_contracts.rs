//! Host instantiation of the shared decomposition conformance clauses.
//!
//! Leto joins the role trait per ADR 0039 section 3: the same clause
//! suite the GPU backends run executes against the CPU reference pair.

use hephaestus_conformance::assert_decomposition_contract;
use hephaestus_host::{HostDecompositionOps, HostDevice};

#[test]
fn host_satisfies_the_decomposition_contract() {
    assert_decomposition_contract(&HostDevice::new(), &HostDecompositionOps);
}
