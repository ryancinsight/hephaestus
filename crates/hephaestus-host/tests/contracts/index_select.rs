//! Host instantiation of the shared index-selection conformance clauses.
//!
//! There is no `leto::index_select` to delegate to, so `HostIndexSelectOps`
//! carries the direct implementation and these clauses pin its value
//! semantics with hand-computed fixtures every accelerator's differential
//! test must match.

use hephaestus_conformance::assert_index_select_contract;
use hephaestus_host::{HostDevice, HostIndexSelectOps};

#[test]
fn host_satisfies_the_index_select_contract() {
    assert_index_select_contract::<_, _, i32>(&HostDevice::new(), &HostIndexSelectOps);
}
