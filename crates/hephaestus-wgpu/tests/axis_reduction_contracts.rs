//! WGPU instantiation of the shared axis-reduction conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`ComputeDevice`] and [`AxisReductionOps`]; this file only supplies the
//! device and the backend's seam value. That is the shape ADR 0038 specifies for
//! every backend: a contract clause is written once and executed by all of them,
//! rather than hand-written four times and allowed to drift.
//!
//! The clauses cover `prod_axis_into` and `prepare_reduce_axis_into`, two of the
//! six shared entry points that no backend exercised before `ATLAS-ARCH-001`.

use hephaestus_conformance::assert_axis_reduction_contract;
use hephaestus_wgpu::{WgpuAxisReductionOps, WgpuDevice};

fn device_or_skip() -> Option<WgpuDevice> {
    static DEVICE: std::sync::OnceLock<Option<WgpuDevice>> = std::sync::OnceLock::new();
    DEVICE
        .get_or_init(
            || match WgpuDevice::try_default("hephaestus-axis-reduction-test") {
                Ok(device) => Some(device),
                Err(e) => {
                    eprintln!("skipping wgpu axis-reduction test: {e}");
                    None
                }
            },
        )
        .clone()
}

#[test]
fn wgpu_satisfies_the_axis_reduction_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_axis_reduction_contract(&device, &WgpuAxisReductionOps);
}
