//! Host instantiation of the shared stacking conformance clauses.
//!
//! `HostStackOps` delegates to `leto::stack` directly (ADR 0046 §5's CPU
//! reference pair), so this suite doubles as the oracle every accelerator's
//! differential test compares against.

use hephaestus_conformance::assert_stack_contract;
use hephaestus_host::{HostDevice, HostStackOps};

#[test]
fn host_satisfies_the_stack_contract() {
    assert_stack_contract::<_, _, i32>(&HostDevice::new(), &HostStackOps);
}
