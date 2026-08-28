use hephaestus_core::{FftDirection, FftOps};
use leto::Layout;

use super::{
    conformance::{TestScalar, download, operands},
    required_device_or_skip,
};
use crate::{ComputeDevice, WgpuDevice, WgpuFftOps};

fn forward<T: TestScalar>(
    device: &WgpuDevice,
    real_host: [f32; 2],
    imaginary_host: [f32; 2],
) -> (Vec<T>, Vec<T>) {
    let real = device
        .upload(&real_host.map(T::from_input))
        .expect("special-value real upload");
    let imaginary = device
        .upload(&imaginary_host.map(T::from_input))
        .expect("special-value imaginary upload");
    let layout = Layout::c_contiguous([2]).expect("dense special-value layout");
    let prepared = WgpuFftOps
        .prepare_fft(
            device,
            operands(&real, &imaginary, &layout),
            FftDirection::Forward,
        )
        .unwrap_or_else(|error| panic!("{} special-value preparation failed: {error}", T::LABEL));
    WgpuFftOps
        .dispatch_fft(device, &prepared)
        .unwrap_or_else(|error| panic!("{} special-value dispatch failed: {error}", T::LABEL));
    (download(device, &real), download(device, &imaginary))
}

fn assert_special_values<T: TestScalar>(device: &WgpuDevice) {
    let (nan_real, _) = forward::<T>(device, [f32::NAN, 0.0], [0.0, 0.0]);
    assert!(nan_real.into_iter().all(|value| value.to_output().is_nan()));

    let (infinite_real, _) = forward::<T>(device, [f32::INFINITY, 0.0], [0.0, 0.0]);
    assert!(
        infinite_real
            .into_iter()
            .all(|value| !value.to_output().is_finite())
    );

    let (zero_real, zero_imaginary) = forward::<T>(device, [-0.0, 0.0], [-0.0, 0.0]);
    assert!(zero_real.into_iter().all(|value| value.to_output() == 0.0));
    assert!(
        zero_imaginary
            .into_iter()
            .all(|value| value.to_output() == 0.0)
    );
}

#[test]
fn scalar_widths_follow_the_documented_special_value_contract() {
    let Some(device) = required_device_or_skip() else {
        return;
    };
    assert_special_values::<f32>(&device);
    assert_special_values::<eunomia::F16>(&device);
}
