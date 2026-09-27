//! Host instantiation of the shared padding conformance clauses.
//!
//! `HostPadOps` delegates to `leto::pad` directly (ADR 0046 §5's CPU
//! reference pair), so this suite doubles as the oracle every accelerator's
//! differential test compares against.

use hephaestus_conformance::assert_pad_contract;
use hephaestus_host::{HostDevice, HostPadOps};

#[test]
fn host_satisfies_the_pad_contract() {
    assert_pad_contract::<_, _, i32>(&HostDevice::new(), &HostPadOps);
}
