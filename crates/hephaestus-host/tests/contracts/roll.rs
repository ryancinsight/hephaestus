//! Host instantiation of the shared circular-shift conformance clauses.
//!
//! There is no `leto::roll` to delegate to, so `HostRollOps` carries the
//! direct implementation and these clauses pin its value semantics with
//! hand-computed fixtures every accelerator's differential test must
//! match.

use hephaestus_conformance::assert_roll_contract;
use hephaestus_host::{HostDevice, HostRollOps};

#[test]
fn host_satisfies_the_roll_contract() {
    assert_roll_contract::<_, _, i32>(&HostDevice::new(), &HostRollOps);
}
