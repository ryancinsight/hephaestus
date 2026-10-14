//! Host instantiation of the shared conditional-selection conformance clauses.
//!
//! There is no `leto::where_cond` to delegate to, so `HostWhereOps`
//! carries the direct implementation and these clauses pin its value
//! semantics with hand-computed fixtures every accelerator's differential
//! test must match.

use hephaestus_conformance::{assert_where_contract, assert_where_special_cond_values};
use hephaestus_host::{HostDevice, HostWhereOps};

#[test]
fn host_satisfies_the_where_contract() {
    let device = HostDevice::new();
    assert_where_contract::<_, _, i32>(&device, &HostWhereOps);
    assert_where_contract::<_, _, f32>(&device, &HostWhereOps);
    assert_where_special_cond_values(&device, &HostWhereOps);
}
