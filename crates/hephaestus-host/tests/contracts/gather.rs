//! Host instantiation of the shared gather conformance clauses.
//!
//! There is no `leto::gather` to delegate to, so `HostGatherOps` carries
//! the direct implementation and these clauses pin its value semantics
//! with hand-computed fixtures every accelerator's differential test must
//! match.

use hephaestus_conformance::assert_gather_contract;
use hephaestus_host::{HostDevice, HostGatherOps};

#[test]
fn host_satisfies_the_gather_contract() {
    assert_gather_contract::<_, _, i32>(&HostDevice::new(), &HostGatherOps);
}
