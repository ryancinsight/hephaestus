//! Host instantiation of the shared inner-product conformance clauses.
//!
//! There is no strided `leto::dot` to delegate to, so `HostDotOps`
//! carries the direct implementation and these clauses pin its value
//! semantics with hand-computed fixtures every accelerator's
//! differential test must match.

use hephaestus_conformance::{assert_dot_contract, assert_dot_special_float_values};
use hephaestus_host::{HostDevice, HostDotOps};

#[test]
fn host_satisfies_the_dot_contract() {
    let device = HostDevice::new();
    assert_dot_contract::<_, _, i32>(&device, &HostDotOps);
    assert_dot_contract::<_, _, f32>(&device, &HostDotOps);
    assert_dot_special_float_values(&device, &HostDotOps);
}
