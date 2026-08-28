use hephaestus_core::{FftDirection, FftOps};
use leto::Layout;

use super::{
    conformance::{
        TestScalar, axis_rounding_sites, download, host_input, operands, relative_error_bound,
        relative_l2_error,
    },
    required_device_or_skip,
};
use crate::{ComputeDevice, WgpuDevice, WgpuFftOps};

fn expected_pairs(real: &[f32], imaginary: &[f32]) -> Vec<[f64; 2]> {
    real.iter()
        .zip(imaginary)
        .map(|(&real, &imaginary)| [f64::from(real), f64::from(imaginary)])
        .collect()
}

fn roundtrip_sites<const R: usize>(shape: [usize; R]) -> usize {
    1 + shape
        .into_iter()
        .map(|length| 2 * axis_rounding_sites(length) + usize::from(length > 1))
        .sum::<usize>()
}

fn assert_roundtrip<T: TestScalar, const R: usize>(device: &WgpuDevice, shape: [usize; R]) {
    let (real_host, imaginary_host) = host_input(shape);
    assert_roundtrip_input::<T, R>(device, shape, &real_host, &imaginary_host);
}

fn assert_roundtrip_input<T: TestScalar, const R: usize>(
    device: &WgpuDevice,
    shape: [usize; R],
    real_host: &[f32],
    imaginary_host: &[f32],
) {
    let real_storage = real_host
        .iter()
        .copied()
        .map(T::from_input)
        .collect::<Vec<_>>();
    let imaginary_storage = imaginary_host
        .iter()
        .copied()
        .map(T::from_input)
        .collect::<Vec<_>>();
    let real = device
        .upload(&real_storage)
        .expect("round-trip real upload");
    let imaginary = device
        .upload(&imaginary_storage)
        .expect("round-trip imaginary upload");
    let layout = Layout::c_contiguous(shape).expect("dense round-trip layout");
    let forward = WgpuFftOps
        .prepare_fft(
            device,
            operands(&real, &imaginary, &layout),
            FftDirection::Forward,
        )
        .unwrap_or_else(|error| panic!("{} forward preparation failed: {error}", T::LABEL));
    let inverse = WgpuFftOps
        .prepare_fft(
            device,
            operands(&real, &imaginary, &layout),
            FftDirection::Inverse,
        )
        .unwrap_or_else(|error| panic!("{} inverse preparation failed: {error}", T::LABEL));
    WgpuFftOps
        .dispatch_fft(device, &forward)
        .unwrap_or_else(|error| panic!("{} forward dispatch failed: {error}", T::LABEL));
    WgpuFftOps
        .dispatch_fft(device, &inverse)
        .unwrap_or_else(|error| panic!("{} inverse dispatch failed: {error}", T::LABEL));

    let actual_real = download(device, &real);
    let actual_imaginary = download(device, &imaginary);
    let expected = expected_pairs(real_host, imaginary_host);
    let relative_error = relative_l2_error(&actual_real, &actual_imaginary, &expected);
    let bound = relative_error_bound::<T>(roundtrip_sites(shape));
    assert!(
        relative_error <= bound,
        "{} round-trip relative L2 error {relative_error:.3e} exceeds derived bound {bound:.3e}",
        T::LABEL
    );
}

fn assert_staged_mode<T: TestScalar>(device: &WgpuDevice) {
    const LENGTH: usize = 4096;
    const MODE: usize = 17;
    const AMPLITUDE: f32 = 0.125;
    let (real_host, imaginary_host): (Vec<_>, Vec<_>) = (0..LENGTH)
        .map(|index| {
            let phase = core::f32::consts::TAU * (MODE * index) as f32 / LENGTH as f32;
            (AMPLITUDE * phase.cos(), AMPLITUDE * phase.sin())
        })
        .unzip();
    let real_storage = real_host
        .iter()
        .copied()
        .map(T::from_input)
        .collect::<Vec<_>>();
    let imaginary_storage = imaginary_host
        .iter()
        .copied()
        .map(T::from_input)
        .collect::<Vec<_>>();
    let real = device
        .upload(&real_storage)
        .expect("staged mode real upload");
    let imaginary = device
        .upload(&imaginary_storage)
        .expect("staged mode imaginary upload");
    let layout = Layout::c_contiguous([LENGTH]).expect("dense staged mode layout");
    let forward = WgpuFftOps
        .prepare_fft(
            device,
            operands(&real, &imaginary, &layout),
            FftDirection::Forward,
        )
        .unwrap_or_else(|error| panic!("{} staged preparation failed: {error}", T::LABEL));
    WgpuFftOps
        .dispatch_fft(device, &forward)
        .unwrap_or_else(|error| panic!("{} staged dispatch failed: {error}", T::LABEL));

    let mut expected = vec![[0.0, 0.0]; LENGTH];
    expected[MODE] = [f64::from(AMPLITUDE) * LENGTH as f64, 0.0];
    let actual_real = download(device, &real);
    let actual_imaginary = download(device, &imaginary);
    let relative_error = relative_l2_error(&actual_real, &actual_imaginary, &expected);
    let bound = relative_error_bound::<T>(1 + axis_rounding_sites(LENGTH));
    assert!(
        relative_error <= bound,
        "{} staged-mode relative L2 error {relative_error:.3e} exceeds derived bound {bound:.3e}",
        T::LABEL
    );

    assert_roundtrip_input::<T, 1>(device, [LENGTH], &real_host, &imaginary_host);
}

#[test]
fn scalar_widths_round_trip_across_ranks_and_staged_execution() {
    let Some(device) = required_device_or_skip() else {
        return;
    };
    assert_roundtrip::<f32, 1>(&device, [8]);
    assert_roundtrip::<eunomia::F16, 1>(&device, [8]);
    assert_roundtrip::<f32, 1>(&device, [3]);
    assert_roundtrip::<eunomia::F16, 1>(&device, [3]);
    assert_roundtrip::<f32, 2>(&device, [2, 3]);
    assert_roundtrip::<eunomia::F16, 2>(&device, [2, 3]);
    assert_roundtrip::<f32, 3>(&device, [2, 2, 3]);
    assert_roundtrip::<eunomia::F16, 3>(&device, [2, 2, 3]);
    assert_staged_mode::<f32>(&device);
    assert_staged_mode::<eunomia::F16>(&device);
}

#[test]
fn normwise_oracle_rejects_zero_output() {
    let actual_real = [0.0_f32, 0.0];
    let actual_imaginary = [0.0_f32, 0.0];
    let expected = [[0.5, -0.25], [-1.0, 0.75]];
    let error = relative_l2_error(&actual_real, &actual_imaginary, &expected);
    assert_eq!(error, 1.0);
    assert!(error > relative_error_bound::<eunomia::F16>(265));
}
