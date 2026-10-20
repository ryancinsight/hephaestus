//! Per-operator accuracy sweeps (HEPH-KERNEL-TAIL-ACCURACY) on WGSL.
//!
//! The backend is the process device (`WGPU_BACKEND` selects Vulkan or DX12;
//! run once per backend). Each test sweeps one rendering over its failing
//! range plus structural edges against the `f64` value-function reference.

use hephaestus_conformance::{
    SweepDialect, assert_celu_sweep, assert_elu_sweep, assert_erfc_sweep, assert_expm1_sweep,
    assert_gelu_grad_sweep, assert_gelu_sweep, assert_gelu_tanh_grad_sweep, assert_gelu_tanh_sweep,
    assert_log1p_sweep, assert_mish_grad_sweep, assert_mish_sweep, assert_sigmoid_sweep,
    assert_silu_grad_sweep, assert_silu_sweep, assert_softplus_grad_sweep, assert_softplus_sweep,
};
use hephaestus_wgpu::{WgpuDevice, WgpuElementwiseOps, WgpuParameterizedUnaryOps};

fn device_or_skip() -> Option<WgpuDevice> {
    super::device_or_skip()
}

#[test]
fn expm1_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_expm1_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}

#[test]
fn log1p_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_log1p_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}

#[test]
fn elu_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_elu_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}

#[test]
fn celu_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_celu_sweep(&device, &WgpuParameterizedUnaryOps, SweepDialect::Wgsl);
}

#[test]
fn softplus_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_softplus_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}

#[test]
fn sigmoid_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_sigmoid_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}

#[test]
fn softplus_grad_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_softplus_grad_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}

#[test]
fn erfc_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_erfc_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}

#[test]
fn gelu_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_gelu_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}

#[test]
fn gelu_grad_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_gelu_grad_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}

#[test]
fn silu_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_silu_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}

#[test]
fn silu_grad_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_silu_grad_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}

#[test]
fn mish_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_mish_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}

#[test]
fn mish_grad_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_mish_grad_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}

#[test]
fn gelu_tanh_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_gelu_tanh_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}

#[test]
fn gelu_tanh_grad_accuracy_sweep() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_gelu_tanh_grad_sweep(&device, &WgpuElementwiseOps, SweepDialect::Wgsl);
}
