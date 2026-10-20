//! Per-operator accuracy sweeps (HEPH-KERNEL-TAIL-ACCURACY) on Metal.
//!
//! Metal authors WGSL through its native device, so these assert the WGSL
//! bound family. Each test sweeps one rendering over its failing range plus
//! structural edges against the `f64` value-function reference. On a host
//! without a Metal device each test skips; hardware CI sets
//! `HEPHAESTUS_METAL_REQUIRE_DEVICE=1` so acquisition failure fails the lane.

#![cfg(target_os = "macos")]
use hephaestus_conformance::{
    SweepDialect, assert_celu_sweep, assert_elu_sweep, assert_erfc_sweep, assert_expm1_sweep,
    assert_gelu_grad_sweep, assert_gelu_sweep, assert_gelu_tanh_grad_sweep, assert_gelu_tanh_sweep,
    assert_log1p_sweep, assert_mish_grad_sweep, assert_mish_sweep, assert_sigmoid_sweep,
    assert_silu_grad_sweep, assert_silu_sweep, assert_softplus_grad_sweep, assert_softplus_sweep,
};
use hephaestus_metal::{MetalDevice, MetalElementwiseOps, MetalParameterizedUnaryOps};

/// Acquire a device, or `None` to skip (no Metal device).
fn device(test: &str) -> Option<MetalDevice> {
    match MetalDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) => {
            if std::env::var_os("HEPHAESTUS_METAL_REQUIRE_DEVICE").is_some() {
                panic!("Metal device required for {test}, but acquisition failed: {error}");
            }
            eprintln!("skip {test}: Metal device unavailable ({error})");
            None
        }
    }
}

#[test]
fn expm1_accuracy_sweep() {
    let Some(device) = device("expm1 sweep") else {
        return;
    };
    assert_expm1_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}

#[test]
fn log1p_accuracy_sweep() {
    let Some(device) = device("log1p sweep") else {
        return;
    };
    assert_log1p_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}

#[test]
fn elu_accuracy_sweep() {
    let Some(device) = device("elu sweep") else {
        return;
    };
    assert_elu_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}

#[test]
fn celu_accuracy_sweep() {
    let Some(device) = device("celu sweep") else {
        return;
    };
    assert_celu_sweep(&device, &MetalParameterizedUnaryOps, SweepDialect::Wgsl);
}

#[test]
fn softplus_accuracy_sweep() {
    let Some(device) = device("softplus sweep") else {
        return;
    };
    assert_softplus_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}

#[test]
fn sigmoid_accuracy_sweep() {
    let Some(device) = device("sigmoid sweep") else {
        return;
    };
    assert_sigmoid_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}

#[test]
fn softplus_grad_accuracy_sweep() {
    let Some(device) = device("softplus-grad sweep") else {
        return;
    };
    assert_softplus_grad_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}

#[test]
fn erfc_accuracy_sweep() {
    let Some(device) = device("erfc sweep") else {
        return;
    };
    assert_erfc_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}

#[test]
fn gelu_accuracy_sweep() {
    let Some(device) = device("gelu sweep") else {
        return;
    };
    assert_gelu_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}

#[test]
fn gelu_grad_accuracy_sweep() {
    let Some(device) = device("gelu-grad sweep") else {
        return;
    };
    assert_gelu_grad_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}

#[test]
fn silu_accuracy_sweep() {
    let Some(device) = device("silu sweep") else {
        return;
    };
    assert_silu_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}

#[test]
fn silu_grad_accuracy_sweep() {
    let Some(device) = device("silu-grad sweep") else {
        return;
    };
    assert_silu_grad_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}

#[test]
fn mish_accuracy_sweep() {
    let Some(device) = device("mish sweep") else {
        return;
    };
    assert_mish_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}

#[test]
fn mish_grad_accuracy_sweep() {
    let Some(device) = device("mish-grad sweep") else {
        return;
    };
    assert_mish_grad_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}

#[test]
fn gelu_tanh_accuracy_sweep() {
    let Some(device) = device("gelu-tanh sweep") else {
        return;
    };
    assert_gelu_tanh_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}

#[test]
fn gelu_tanh_grad_accuracy_sweep() {
    let Some(device) = device("gelu-tanh-grad sweep") else {
        return;
    };
    assert_gelu_tanh_grad_sweep(&device, &MetalElementwiseOps::default(), SweepDialect::Wgsl);
}
