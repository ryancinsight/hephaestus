//! Host instantiation of the shared stable-sort conformance clauses.
//!
//! There is no `leto::sort` to delegate to, so `HostSortOps` carries the
//! direct implementation and these clauses pin its value semantics with
//! hand-computed fixtures every accelerator's differential test must
//! match.

use hephaestus_conformance::{assert_sort_contract, assert_sort_special_float_values};
use hephaestus_host::{HostDevice, HostSortOps};

#[test]
fn host_satisfies_the_sort_contract() {
    let device = HostDevice::new();
    assert_sort_contract::<_, _, i32>(&device, &HostSortOps);
    assert_sort_contract::<_, _, f32>(&device, &HostSortOps);
    assert_sort_special_float_values(&device, &HostSortOps);
}
