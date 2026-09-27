//! WGPU instantiation of the shared top-k conformance clause.

use hephaestus_conformance::assert_topk_contract;
use hephaestus_wgpu::WgpuTopKOps;

pub(super) fn wgpu_satisfies_the_topk_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_topk_contract(&device, &WgpuTopKOps);
}
