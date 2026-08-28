use std::sync::OnceLock;

use eunomia::F16;
use hephaestus_core::{
    DeviceFeature, DevicePreference, FftDirection, FftOps, GroupedCommandStream, HephaestusError,
    KernelDevice, KernelSource,
};
use leto::Layout;

use super::super::kernel::{
    Butterfly, ChirpKernel, ChirpScale, FftKernel, FusedKernel, Pack, PackKernel,
};
use super::WgpuFftOps;
use crate::{ComputeDevice, WgpuDevice};

#[path = "scalar_tests/conformance.rs"]
mod conformance;
#[path = "scalar_tests/roundtrip.rs"]
mod roundtrip;
#[path = "scalar_tests/special_values.rs"]
mod special_values;

use conformance::{download, host_input, operands, relative_error_bound, relative_l2_error};

fn required_device_or_skip() -> Option<WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            match WgpuDevice::try_with_device_preference_and_required_device_features(
                "hephaestus-fft-half-test",
                DevicePreference::HighPerformance,
                &[DeviceFeature::ShaderF16],
            ) {
                Ok(device) => Some(device),
                Err(error @ HephaestusError::AdapterUnavailable { .. }) => {
                    if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_some() {
                        panic!("WGPU ShaderF16 adapter required, but acquisition failed: {error}");
                    }
                    eprintln!("skipping binary16 FFT test: ShaderF16 adapter unavailable");
                    None
                }
                Err(error) => panic!("binary16 FFT tests require a working provider: {error}"),
            }
        })
        .clone()
}

fn default_device_or_skip() -> Option<WgpuDevice> {
    match WgpuDevice::try_default("hephaestus-fft-half-rejection-test") {
        Ok(device) => Some(device),
        Err(error @ HephaestusError::AdapterUnavailable { .. }) => {
            if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_some() {
                panic!("WGPU adapter required, but acquisition failed: {error}");
            }
            None
        }
        Err(error) => panic!("binary16 FFT rejection test requires a working provider: {error}"),
    }
}

#[test]
fn half_precision_sources_contain_no_wider_arithmetic() {
    let sources = [
        FftKernel::<F16, Butterfly>::new().source(),
        FusedKernel::<F16>::new().source(),
        PackKernel::<F16, Pack>::new().source(),
        ChirpKernel::<F16, ChirpScale>::new().source(),
    ];
    for source in sources {
        assert!(source.starts_with("enable f16;"));
        assert!(!source.contains("{{scalar}}"));
        assert!(!source.contains("f32"));
    }
}

#[test]
fn half_precision_bluestein_roundtrip_obeys_derived_bound() {
    let Some(device) = required_device_or_skip() else {
        return;
    };
    let shape = [3, 3, 3];
    let (real_host, imaginary_host) = host_input(shape);
    let real = device
        .upload(
            &real_host
                .iter()
                .copied()
                .map(F16::from_f32)
                .collect::<Vec<_>>(),
        )
        .expect("binary16 real upload");
    let imaginary = device
        .upload(
            &imaginary_host
                .iter()
                .copied()
                .map(F16::from_f32)
                .collect::<Vec<_>>(),
        )
        .expect("binary16 imaginary upload");
    let layout = Layout::c_contiguous(shape).expect("dense binary16 layout");
    let ops = WgpuFftOps;
    let forward = ops
        .prepare_fft(
            &device,
            operands(&real, &imaginary, &layout),
            FftDirection::Forward,
        )
        .expect("binary16 forward preparation");
    let inverse = ops
        .prepare_fft(
            &device,
            operands(&real, &imaginary, &layout),
            FftDirection::Inverse,
        )
        .expect("binary16 inverse preparation");
    ops.dispatch_fft(&device, &forward)
        .expect("binary16 forward dispatch");
    ops.dispatch_fft(&device, &inverse)
        .expect("binary16 inverse dispatch");

    let sites = 1 + 3 * (43 + 45);
    let expected = real_host
        .iter()
        .zip(&imaginary_host)
        .map(|(&real, &imaginary)| [f64::from(real), f64::from(imaginary)])
        .collect::<Vec<_>>();
    let error = relative_l2_error(
        &download(&device, &real),
        &download(&device, &imaginary),
        &expected,
    );
    let bound = relative_error_bound::<F16>(sites);
    assert!(
        error <= bound,
        "binary16 reconstruction relative L2 error {error:.3e} exceeds derived bound {bound:.3e}"
    );
}

#[test]
fn half_precision_warm_grouped_dispatch_reuses_prepared_resources() {
    let Some(device) = required_device_or_skip() else {
        return;
    };
    let shape = [3];
    let (real_host, imaginary_host) = host_input(shape);
    let real_storage = real_host
        .iter()
        .copied()
        .map(F16::from_f32)
        .collect::<Vec<_>>();
    let imaginary_storage = imaginary_host
        .iter()
        .copied()
        .map(F16::from_f32)
        .collect::<Vec<_>>();
    let real = device.upload(&real_storage).expect("binary16 real upload");
    let imaginary = device
        .upload(&imaginary_storage)
        .expect("binary16 imaginary upload");
    let layout = Layout::c_contiguous(shape).expect("dense binary16 layout");
    let forward = WgpuFftOps
        .prepare_fft(
            &device,
            operands(&real, &imaginary, &layout),
            FftDirection::Forward,
        )
        .expect("binary16 forward preparation");
    let inverse = WgpuFftOps
        .prepare_fft(
            &device,
            operands(&real, &imaginary, &layout),
            FftDirection::Inverse,
        )
        .expect("binary16 inverse preparation");
    let forward_commands = forward.plan.commands.as_ptr();
    let inverse_commands = inverse.plan.commands.as_ptr();
    let forward_workspace = forward
        .plan
        .workspace
        .as_ref()
        .expect("Bluestein forward workspace")
        .real
        .raw()
        .clone();
    let inverse_workspace = inverse
        .plan
        .workspace
        .as_ref()
        .expect("Bluestein inverse workspace")
        .real
        .raw()
        .clone();
    let forward_radix_twiddle = forward
        .plan
        .radix_twiddle
        .as_ref()
        .expect("Bluestein forward radix twiddle")
        .raw()
        .clone();
    let inverse_radix_twiddle = inverse
        .plan
        .radix_twiddle
        .as_ref()
        .expect("Bluestein inverse radix twiddle")
        .raw()
        .clone();

    let mut stream = device.stream().expect("binary16 grouped stream");
    stream
        .encode_grouped_sequence("hephaestus-fft-half-warm-pass", |sequence| {
            forward.encode_in_sequence(sequence)?;
            inverse.encode_in_sequence(sequence)
        })
        .expect("binary16 grouped encoding");
    stream
        .submit_with_timeout(std::time::Duration::from_secs(10))
        .expect("bounded binary16 grouped submission");

    assert_eq!(forward.plan.commands.as_ptr(), forward_commands);
    assert_eq!(inverse.plan.commands.as_ptr(), inverse_commands);
    assert_eq!(
        forward
            .plan
            .workspace
            .as_ref()
            .expect("Bluestein forward workspace")
            .real
            .raw(),
        &forward_workspace
    );
    assert_eq!(
        inverse
            .plan
            .workspace
            .as_ref()
            .expect("Bluestein inverse workspace")
            .real
            .raw(),
        &inverse_workspace
    );
    assert_eq!(
        forward
            .plan
            .radix_twiddle
            .as_ref()
            .expect("Bluestein forward radix twiddle")
            .raw(),
        &forward_radix_twiddle
    );
    assert_eq!(
        inverse
            .plan
            .radix_twiddle
            .as_ref()
            .expect("Bluestein inverse radix twiddle")
            .raw(),
        &inverse_radix_twiddle
    );
    let expected = real_host
        .iter()
        .zip(&imaginary_host)
        .map(|(&real, &imaginary)| [f64::from(real), f64::from(imaginary)])
        .collect::<Vec<_>>();
    let error = relative_l2_error(
        &download(&device, &real),
        &download(&device, &imaginary),
        &expected,
    );
    let bound = relative_error_bound::<F16>(1 + 43 + 45);
    assert!(
        error <= bound,
        "binary16 warm reconstruction relative L2 error {error:.3e} exceeds derived bound {bound:.3e}"
    );
}

#[test]
fn half_precision_rejects_missing_capability_without_mutation() {
    let Some(device) = default_device_or_skip() else {
        return;
    };
    assert!(!device.supports_device_feature(DeviceFeature::ShaderF16));
    let initial_real = [F16::from_f32(0.5), F16::from_f32(-1.25)];
    let initial_imaginary = [F16::from_f32(-0.25), F16::from_f32(2.0)];
    let real = device.upload(&initial_real).expect("binary16 real upload");
    let imaginary = device
        .upload(&initial_imaginary)
        .expect("binary16 imaginary upload");
    let layout = Layout::c_contiguous([2]).expect("dense binary16 layout");
    let Err(error) = WgpuFftOps.prepare_fft(
        &device,
        operands(&real, &imaginary, &layout),
        FftDirection::Forward,
    ) else {
        panic!("missing ShaderF16 must reject preparation");
    };
    assert_eq!(
        error.to_string(),
        "invalid configuration: WGPU FFT requires the ShaderF16 device feature for binary16"
    );
    assert_eq!(
        download(&device, &real)
            .into_iter()
            .map(F16::to_bits)
            .collect::<Vec<_>>(),
        initial_real.map(F16::to_bits)
    );
    assert_eq!(
        download(&device, &imaginary)
            .into_iter()
            .map(F16::to_bits)
            .collect::<Vec<_>>(),
        initial_imaginary.map(F16::to_bits)
    );
}

#[test]
fn half_precision_rejects_cross_device_dispatch_without_mutation() {
    let Some(device) = required_device_or_skip() else {
        return;
    };
    let other = WgpuDevice::try_with_device_preference_and_required_device_features(
        "hephaestus-fft-half-cross-device-test",
        DevicePreference::HighPerformance,
        &[DeviceFeature::ShaderF16],
    )
    .expect("second ShaderF16 device acquisition");
    let initial_real = [F16::from_f32(0.5), F16::from_f32(-1.25)];
    let initial_imaginary = [F16::from_f32(-0.25), F16::from_f32(2.0)];
    let real = device.upload(&initial_real).expect("binary16 real upload");
    let imaginary = device
        .upload(&initial_imaginary)
        .expect("binary16 imaginary upload");
    let layout = Layout::c_contiguous([2]).expect("dense binary16 layout");
    let ops = WgpuFftOps;
    let prepared = ops
        .prepare_fft(
            &device,
            operands(&real, &imaginary, &layout),
            FftDirection::Forward,
        )
        .expect("binary16 FFT preparation");
    let error = ops
        .dispatch_fft(&other, &prepared)
        .expect_err("cross-device binary16 dispatch must fail");
    assert_eq!(
        error.to_string(),
        "kernel dispatch failed: prepared WGPU FFT belongs to a different device"
    );
    assert_eq!(
        download(&device, &real)
            .into_iter()
            .map(F16::to_bits)
            .collect::<Vec<_>>(),
        initial_real.map(F16::to_bits)
    );
    assert_eq!(
        download(&device, &imaginary)
            .into_iter()
            .map(F16::to_bits)
            .collect::<Vec<_>>(),
        initial_imaginary.map(F16::to_bits)
    );
}
