use bytemuck::Pod;
use eunomia::F16;
use hephaestus_core::{FftDirection, FftOperands, FftOps, StridedView};
use leto::Layout;

use super::required_device_or_skip;
use crate::{
    ComputeDevice, DeviceBuffer, WgpuBuffer, WgpuDevice, WgpuFftOps, WgpuFftScalar, WgpuPreparedFft,
};

const HALF_UNIT_ROUNDOFF: f64 = 1.0 / 2048.0;

pub(super) trait TestScalar: WgpuFftScalar + Pod + Copy {
    const LABEL: &'static str;
    const UNIT_ROUNDOFF: f64;

    fn from_input(value: f32) -> Self;
    fn to_output(self) -> f32;
}

impl TestScalar for f32 {
    const LABEL: &'static str = "binary32";
    const UNIT_ROUNDOFF: f64 = f32::EPSILON as f64 / 2.0;

    fn from_input(value: f32) -> Self {
        value
    }

    fn to_output(self) -> f32 {
        self
    }
}

impl TestScalar for F16 {
    const LABEL: &'static str = "binary16";
    const UNIT_ROUNDOFF: f64 = HALF_UNIT_ROUNDOFF;

    fn from_input(value: f32) -> Self {
        Self::from_f32(value)
    }

    fn to_output(self) -> f32 {
        self.to_f32()
    }
}

pub(super) fn operands<'a, T: TestScalar, const R: usize>(
    real: &'a WgpuBuffer<T>,
    imaginary: &'a WgpuBuffer<T>,
    layout: &'a Layout<R>,
) -> FftOperands<'a, WgpuBuffer<T>, R> {
    FftOperands {
        real: StridedView::new(real, layout),
        imaginary: StridedView::new(imaginary, layout),
    }
}

fn coordinate<const R: usize>(mut linear: usize, shape: [usize; R]) -> [usize; R] {
    let mut coordinate = [0; R];
    for axis in (0..R).rev() {
        coordinate[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    coordinate
}

fn direct_forward<const R: usize>(
    shape: [usize; R],
    real: &[f32],
    imaginary: &[f32],
) -> Vec<[f64; 2]> {
    let elements = shape.into_iter().product();
    (0..elements)
        .map(|output_index| {
            let output_coordinate = coordinate(output_index, shape);
            (0..elements).fold([0.0, 0.0], |sum, input_index| {
                let input_coordinate = coordinate(input_index, shape);
                let phase = (0..R).fold(0.0, |phase, axis| {
                    phase
                        + input_coordinate[axis] as f64 * output_coordinate[axis] as f64
                            / shape[axis] as f64
                });
                let angle = -core::f64::consts::TAU * phase;
                let input = [
                    f64::from(real[input_index]),
                    f64::from(imaginary[input_index]),
                ];
                [
                    sum[0] + input[0].mul_add(angle.cos(), -(input[1] * angle.sin())),
                    sum[1] + input[0].mul_add(angle.sin(), input[1] * angle.cos()),
                ]
            })
        })
        .collect()
}

pub(super) fn host_input<const R: usize>(shape: [usize; R]) -> (Vec<f32>, Vec<f32>) {
    let elements = shape.into_iter().product();
    let real = (0..elements)
        .map(|index| {
            let value = index as f32;
            (0.21 * value).sin() + 0.2 * (0.37 * value).cos()
        })
        .collect();
    let imaginary = (0..elements)
        .map(|index| -0.125 + 0.03125 * index as f32)
        .collect();
    (real, imaginary)
}

pub(super) fn download<T: TestScalar>(device: &WgpuDevice, buffer: &WgpuBuffer<T>) -> Vec<T> {
    let mut host = vec![T::from_input(0.0); buffer.len()];
    device
        .download(buffer, &mut host)
        .unwrap_or_else(|error| panic!("{} FFT readback failed: {error}", T::LABEL));
    host
}

pub(super) fn axis_rounding_sites(length: usize) -> usize {
    if length <= 1 {
        0
    } else if length.is_power_of_two() {
        5 * length.trailing_zeros() as usize
    } else {
        let convolution = (2 * length - 1).next_power_of_two();
        // Premultiply (4), two radix transforms (5 sites/stage each), point
        // multiply (4), convolution normalization (1), and postmultiply (4).
        13 + 10 * convolution.trailing_zeros() as usize
    }
}

fn component_error_bound<T: TestScalar>(
    real: &[f32],
    imaginary: &[f32],
    rounding_sites: usize,
) -> f64 {
    relative_error_bound::<T>(rounding_sites)
        * real
            .iter()
            .zip(imaginary)
            .map(|(&real, &imaginary)| f64::from(real).hypot(f64::from(imaginary)))
            .sum::<f64>()
}

pub(super) fn relative_error_bound<T: TestScalar>(rounding_sites: usize) -> f64 {
    let sites = rounding_sites as f64;
    let accumulated = sites * T::UNIT_ROUNDOFF;
    assert!(
        accumulated < 1.0,
        "{} rounding model requires sites * unit roundoff < 1",
        T::LABEL
    );
    accumulated / (1.0 - accumulated)
}

pub(super) fn relative_l2_error<T: TestScalar>(
    actual_real: &[T],
    actual_imaginary: &[T],
    expected: &[[f64; 2]],
) -> f64 {
    assert_eq!(actual_real.len(), actual_imaginary.len());
    assert_eq!(actual_real.len(), expected.len());
    let (error_squared, expected_squared) = actual_real
        .iter()
        .zip(actual_imaginary)
        .zip(expected)
        .fold(
        (0.0, 0.0),
        |(error_squared, expected_squared),
         ((&actual_real, &actual_imaginary), &[expected_real, expected_imaginary])| {
            let real_error = f64::from(actual_real.to_output()) - expected_real;
            let imaginary_error = f64::from(actual_imaginary.to_output()) - expected_imaginary;
            (
                real_error.mul_add(
                    real_error,
                    imaginary_error.mul_add(imaginary_error, error_squared),
                ),
                expected_real.mul_add(
                    expected_real,
                    expected_imaginary.mul_add(expected_imaginary, expected_squared),
                ),
            )
        },
    );
    if expected_squared == 0.0 {
        error_squared.sqrt()
    } else {
        (error_squared / expected_squared).sqrt()
    }
}

fn assert_forward<T: TestScalar, const R: usize>(device: &WgpuDevice, shape: [usize; R]) {
    let (real_host, imaginary_host) = host_input(shape);
    let expected = direct_forward(shape, &real_host, &imaginary_host);
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
        .expect("scalar FFT real upload");
    let imaginary = device
        .upload(&imaginary_storage)
        .expect("scalar FFT imaginary upload");
    let layout = Layout::c_contiguous(shape).expect("dense scalar FFT layout");
    let ops = WgpuFftOps;
    let prepared: WgpuPreparedFft<R, T> = ops
        .prepare_fft(
            device,
            operands(&real, &imaginary, &layout),
            FftDirection::Forward,
        )
        .unwrap_or_else(|error| panic!("{} FFT preparation failed: {error}", T::LABEL));
    ops.dispatch_fft(device, &prepared)
        .unwrap_or_else(|error| panic!("{} FFT dispatch failed: {error}", T::LABEL));

    let actual_real = download(device, &real);
    let actual_imaginary = download(device, &imaginary);
    let sites = 1 + shape.into_iter().map(axis_rounding_sites).sum::<usize>();
    let relative_bound = relative_error_bound::<T>(sites);
    let relative_error = relative_l2_error(&actual_real, &actual_imaginary, &expected);
    assert!(
        relative_error <= relative_bound,
        "{} {shape:?} spectrum relative L2 error {relative_error:.3e} exceeds derived bound {relative_bound:.3e}",
        T::LABEL,
    );
    let component_bound = component_error_bound::<T>(&real_host, &imaginary_host, sites);
    for (index, ((real, imaginary), expected)) in actual_real
        .into_iter()
        .zip(actual_imaginary)
        .zip(expected)
        .enumerate()
    {
        let error = (f64::from(real.to_output()) - expected[0])
            .hypot(f64::from(imaginary.to_output()) - expected[1]);
        assert!(
            error <= component_bound,
            "{} {shape:?} spectrum error {error:.3e} exceeds derived bound {component_bound:.3e} at {index}",
            T::LABEL,
        );
    }
}

fn assert_staged_impulse<T: TestScalar>(device: &WgpuDevice) {
    const LENGTH: usize = 4096;
    let zero = T::from_input(0.0);
    let one = T::from_input(1.0);
    let mut input = vec![zero; LENGTH];
    input[0] = one;
    let real = device
        .upload(&input)
        .unwrap_or_else(|error| panic!("{} impulse upload failed: {error}", T::LABEL));
    let imaginary = device
        .upload(&[zero; LENGTH])
        .unwrap_or_else(|error| panic!("{} zero upload failed: {error}", T::LABEL));
    let layout = Layout::c_contiguous([LENGTH]).expect("dense scalar FFT layout");
    let ops = WgpuFftOps;
    let prepared = ops
        .prepare_fft(
            device,
            operands(&real, &imaginary, &layout),
            FftDirection::Forward,
        )
        .unwrap_or_else(|error| panic!("{} staged preparation failed: {error}", T::LABEL));
    ops.dispatch_fft(device, &prepared)
        .unwrap_or_else(|error| panic!("{} staged dispatch failed: {error}", T::LABEL));
    assert!(
        download(device, &real)
            .into_iter()
            .all(|value| value.to_output() == 1.0)
    );
    assert!(
        download(device, &imaginary)
            .into_iter()
            .all(|value| value.to_output() == 0.0)
    );
}

#[test]
fn scalar_widths_match_direct_oracles_across_ranks_and_strategies() {
    let Some(device) = required_device_or_skip() else {
        return;
    };
    for shape in [[8], [3], [1]] {
        assert_forward::<f32, 1>(&device, shape);
        assert_forward::<F16, 1>(&device, shape);
    }
    for shape in [[2, 4], [2, 3], [1, 2]] {
        assert_forward::<f32, 2>(&device, shape);
        assert_forward::<F16, 2>(&device, shape);
    }
    for shape in [[2, 2, 4], [2, 2, 3], [1, 1, 1]] {
        assert_forward::<f32, 3>(&device, shape);
        assert_forward::<F16, 3>(&device, shape);
    }
    assert_staged_impulse::<f32>(&device);
    assert_staged_impulse::<F16>(&device);
}
