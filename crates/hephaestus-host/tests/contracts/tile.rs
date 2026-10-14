//! Host instantiation of the shared tiling conformance clauses.
//!
//! There is no `leto::tile` to delegate to, so `HostTileOps` carries the
//! direct implementation and these clauses pin its value semantics with
//! hand-computed fixtures every accelerator's differential test must
//! match.

use hephaestus_conformance::assert_tile_contract;
use hephaestus_host::{HostDevice, HostTileOps};

#[test]
fn host_satisfies_the_tile_contract() {
    assert_tile_contract::<_, _, i32>(&HostDevice::new(), &HostTileOps);
}
