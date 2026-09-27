//! Host instantiation of the shared argmax/argmin conformance clauses.

use hephaestus_conformance::{
    assert_arg_reduce_contract, assert_arg_reduce_transposed_view_contract,
};
use hephaestus_host::{HostArgReduceOps, HostDevice};

#[test]
fn host_satisfies_the_arg_reduce_contract() {
    let device = HostDevice::new();
    assert_arg_reduce_contract(&device, &HostArgReduceOps);
    assert_arg_reduce_transposed_view_contract(&device, &HostArgReduceOps);
}
