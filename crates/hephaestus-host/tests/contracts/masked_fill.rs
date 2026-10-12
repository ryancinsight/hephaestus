//! Host instantiation of the shared masked-fill conformance clauses.
//!
//! There is no `leto::masked_fill` to delegate to, so `HostMaskedFillOps`
//! carries the direct implementation and these clauses pin its value
//! semantics with hand-computed fixtures every accelerator's differential
//! test must match.

use hephaestus_conformance::{assert_masked_fill_contract, assert_masked_fill_special_mask_values};
use hephaestus_host::{HostDevice, HostMaskedFillOps};

#[test]
fn host_satisfies_the_masked_fill_contract() {
    let device = HostDevice::new();
    assert_masked_fill_contract::<_, _, i32>(&device, &HostMaskedFillOps);
    assert_masked_fill_contract::<_, _, f32>(&device, &HostMaskedFillOps);
    assert_masked_fill_special_mask_values(&device, &HostMaskedFillOps);
}
