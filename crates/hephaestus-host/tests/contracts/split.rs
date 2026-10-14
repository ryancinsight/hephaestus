//! Host instantiation of the shared splitting conformance clauses.
//!
//! `HostSplitOps` delegates to `leto::split` directly (ADR 0046 §5's CPU
//! reference pair), so this suite doubles as the oracle every accelerator's
//! differential test compares against.

use hephaestus_conformance::assert_split_contract;
use hephaestus_host::{HostDevice, HostSplitOps};

#[test]
fn host_satisfies_the_split_contract() {
    assert_split_contract::<_, _, i32>(&HostDevice::new(), &HostSplitOps);
}
