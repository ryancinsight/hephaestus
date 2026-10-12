//! Host instantiation of the shared stencil, staggered, and fixed-scheme clauses.

use hephaestus_conformance::{
    assert_fixed_fd_3d_contract, assert_staggered_3d_contract, assert_stencil_contract,
};
use hephaestus_host::{HostDevice, HostFixedFdOps, HostStaggeredOps, HostStencilOps};

#[test]
fn host_satisfies_the_stencil_contract() {
    assert_stencil_contract(&HostDevice::new(), &HostStencilOps);
}

#[test]
fn host_satisfies_the_staggered_contract() {
    assert_staggered_3d_contract(&HostDevice::new(), &HostStaggeredOps);
}

#[test]
fn host_satisfies_the_fixed_fd_contract() {
    assert_fixed_fd_3d_contract(&HostDevice::new(), &HostFixedFdOps);
}
