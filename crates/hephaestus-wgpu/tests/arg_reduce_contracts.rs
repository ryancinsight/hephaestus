//! WGPU instantiation of the shared argmax/argmin conformance clauses.

use hephaestus_conformance::{
    assert_arg_reduce_contract, assert_arg_reduce_transposed_view_contract,
};
use hephaestus_wgpu::WgpuArgReduceOps;

pub(super) fn wgpu_satisfies_the_arg_reduce_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_arg_reduce_contract(&device, &WgpuArgReduceOps);
    assert_arg_reduce_transposed_view_contract(&device, &WgpuArgReduceOps);
}
