//! Host instantiation of the shared scatter-assignment conformance clauses.
//!
//! There is no `leto::index_put` to delegate to, so `HostIndexPutOps`
//! carries the direct implementation and these clauses pin its value
//! semantics with hand-computed fixtures every accelerator's differential
//! test must match.

use hephaestus_conformance::assert_index_put_contract;
use hephaestus_host::{HostDevice, HostIndexPutOps};

#[test]
fn host_satisfies_the_index_put_contract() {
    assert_index_put_contract::<_, _, i32>(&HostDevice::new(), &HostIndexPutOps);
}
