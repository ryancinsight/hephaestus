//! Host instantiation of the shared scatter-add conformance clauses.
//!
//! There is no `leto::scatter_add` to delegate to, so `HostScatterOps`
//! carries the direct implementation and these clauses pin its value
//! semantics with hand-computed fixtures every accelerator's differential
//! test must match — including the ascending-source-order accumulation the
//! order clause observes through `f32` rounding.

use hephaestus_conformance::{assert_scatter_contract, assert_scatter_order_contract};
use hephaestus_host::{HostDevice, HostScatterOps};

#[test]
fn host_satisfies_the_scatter_contract() {
    assert_scatter_contract::<_, _, i32>(&HostDevice::new(), &HostScatterOps);
    assert_scatter_order_contract(&HostDevice::new(), &HostScatterOps);
}
