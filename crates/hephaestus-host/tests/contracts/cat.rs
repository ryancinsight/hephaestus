//! Host instantiation of the shared concatenation conformance clauses.
//!
//! `HostCatOps` delegates to `leto::concat` directly (ADR 0046 §5's CPU
//! reference pair), so this suite doubles as the oracle every accelerator's
//! differential test compares against.

use hephaestus_conformance::assert_cat_contract;
use hephaestus_host::{HostCatOps, HostDevice};

#[test]
fn host_satisfies_the_cat_contract() {
    assert_cat_contract::<_, _, i32>(&HostDevice::new(), &HostCatOps);
}
