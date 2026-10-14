//! Host instantiation of the shared nonzero-coordinate conformance clauses.
//!
//! There is no `leto::nonzero` to delegate to, so `HostNonzeroOps`
//! carries the direct implementation and these clauses pin its value
//! semantics with hand-computed fixtures every accelerator's
//! differential test must match.

use hephaestus_conformance::{
    assert_nonzero_contract, assert_nonzero_rejects_aliased_output,
    assert_nonzero_special_float_values,
};
use hephaestus_host::{HostDevice, HostNonzeroOps};

#[test]
fn host_satisfies_the_nonzero_contract() {
    let device = HostDevice::new();
    assert_nonzero_contract::<_, _, i32>(&device, &HostNonzeroOps);
    assert_nonzero_contract::<_, _, f32>(&device, &HostNonzeroOps);
    assert_nonzero_special_float_values(&device, &HostNonzeroOps);
    assert_nonzero_rejects_aliased_output(&device, &HostNonzeroOps);
}
