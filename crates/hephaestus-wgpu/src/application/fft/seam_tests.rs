use std::sync::OnceLock;

use hephaestus_core::{
    FftDirection, FftOperands, FftOps, GroupedCommandStream, HephaestusError, KernelDevice,
    StridedView,
};
use leto::Layout;

use super::{WgpuFftOps, WgpuPreparedFft};
use crate::{ComputeDevice, DeviceBuffer, WgpuBuffer, WgpuDevice};

fn device_or_skip() -> Option<WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE
        .get_or_init(|| match WgpuDevice::try_default("hephaestus-fft-test") {
            Ok(device) => Some(device),
            Err(error @ HephaestusError::AdapterUnavailable { .. }) => {
                if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_some() {
                    panic!("WGPU adapter required, but acquisition failed: {error}");
                }
                eprintln!("skipping wgpu FFT test: adapter unavailable");
                None
            }
            Err(error) => panic!("WGPU FFT tests require a working provider: {error}"),
        })
        .clone()
}

fn operands<'a, const R: usize>(
    real: &'a WgpuBuffer<f32>,
    imaginary: &'a WgpuBuffer<f32>,
    layout: &'a Layout<R>,
) -> FftOperands<'a, WgpuBuffer<f32>, R> {
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

fn exact_u32(value: usize) -> u32 {
    u32::try_from(value).expect("invariant: test dimensions fit u32")
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
                        + f64::from(exact_u32(input_coordinate[axis]))
                            * f64::from(exact_u32(output_coordinate[axis]))
                            / f64::from(exact_u32(shape[axis]))
                });
                let angle = -core::f64::consts::TAU * phase;
                let twiddle = [angle.cos(), angle.sin()];
                let input = [
                    f64::from(real[input_index]),
                    f64::from(imaginary[input_index]),
                ];
                [
                    sum[0] + input[0].mul_add(twiddle[0], -(input[1] * twiddle[1])),
                    sum[1] + input[0].mul_add(twiddle[1], input[1] * twiddle[0]),
                ]
            })
        })
        .collect()
}

fn download(device: &WgpuDevice, buffer: &WgpuBuffer<f32>) -> Vec<f32> {
    let mut host = vec![0.0; buffer.len()];
    device
        .download(buffer, &mut host)
        .expect("invariant: FFT readback succeeds");
    host
}

fn assert_complex_close(
    actual_real: &[f32],
    actual_imaginary: &[f32],
    expected: &[[f64; 2]],
    operation_count: usize,
) {
    assert_eq!(actual_real.len(), expected.len());
    assert_eq!(actual_imaginary.len(), expected.len());
    // Each output accumulates `operation_count` f32 terms. The factor covers
    // the radix/Bluestein depth and single-rounding differences from the f64
    // direct oracle while retaining a bound proportional to machine epsilon.
    let tolerance = 128.0 * f64::from(f32::EPSILON) * f64::from(exact_u32(operation_count));
    for (index, ((&real, &imaginary), expected)) in actual_real
        .iter()
        .zip(actual_imaginary)
        .zip(expected)
        .enumerate()
    {
        let scale = expected[0].abs().max(expected[1].abs()).max(1.0);
        assert!(
            (f64::from(real) - expected[0]).abs() <= tolerance * scale,
            "real[{index}]={real}, expected {}, tolerance {}",
            expected[0],
            tolerance * scale
        );
        assert!(
            (f64::from(imaginary) - expected[1]).abs() <= tolerance * scale,
            "imaginary[{index}]={imaginary}, expected {}, tolerance {}",
            expected[1],
            tolerance * scale
        );
    }
}

fn prepare<'a, const R: usize>(
    ops: &WgpuFftOps,
    device: &'a WgpuDevice,
    real: &'a WgpuBuffer<f32>,
    imaginary: &'a WgpuBuffer<f32>,
    layout: &'a Layout<R>,
    direction: FftDirection,
) -> WgpuPreparedFft<R> {
    ops.prepare_fft(device, operands(real, imaginary, layout), direction)
        .expect("invariant: FFT preparation succeeds")
}

fn verify_shape<const R: usize>(device: &WgpuDevice, shape: [usize; R]) {
    let elements = shape.into_iter().product();
    let real_host = (0..elements)
        .map(|index| 0.25 + f32::from(u16::try_from(index).expect("small test index")) * 0.125)
        .collect::<Vec<_>>();
    let imaginary_host = (0..elements)
        .map(|index| -0.375 + f32::from(u16::try_from(index).expect("small test index")) * 0.0625)
        .collect::<Vec<_>>();
    let expected_forward = direct_forward(shape, &real_host, &imaginary_host);
    let real = device.upload(&real_host).expect("invariant: real upload");
    let imaginary = device
        .upload(&imaginary_host)
        .expect("invariant: imaginary upload");
    let layout = Layout::c_contiguous(shape).expect("invariant: dense test layout");
    let ops = WgpuFftOps;
    let forward = prepare(
        &ops,
        device,
        &real,
        &imaginary,
        &layout,
        FftDirection::Forward,
    );
    let inverse = prepare(
        &ops,
        device,
        &real,
        &imaginary,
        &layout,
        FftDirection::Inverse,
    );

    ops.dispatch_fft(device, &forward)
        .expect("invariant: forward FFT dispatch");
    assert_complex_close(
        &download(device, &real),
        &download(device, &imaginary),
        &expected_forward,
        elements,
    );

    ops.dispatch_fft(device, &inverse)
        .expect("invariant: inverse FFT dispatch");
    let expected_input = real_host
        .iter()
        .zip(&imaginary_host)
        .map(|(&real, &imaginary)| [f64::from(real), f64::from(imaginary)])
        .collect::<Vec<_>>();
    assert_complex_close(
        &download(device, &real),
        &download(device, &imaginary),
        &expected_input,
        elements,
    );

    let mut stream = device.stream().expect("invariant: command stream");
    ops.encode_fft(device, &forward, &mut stream)
        .expect("invariant: composed forward encoding");
    stream
        .submit_with_timeout(std::time::Duration::from_secs(10))
        .expect("invariant: bounded composed forward submission");
    assert_complex_close(
        &download(device, &real),
        &download(device, &imaginary),
        &expected_forward,
        elements,
    );

    let mut stream = device.stream().expect("invariant: command stream");
    ops.encode_fft(device, &inverse, &mut stream)
        .expect("invariant: composed inverse encoding");
    stream
        .submit_with_timeout(std::time::Duration::from_secs(10))
        .expect("invariant: bounded composed inverse submission");
    assert_complex_close(
        &download(device, &real),
        &download(device, &imaginary),
        &expected_input,
        elements,
    );
}

#[test]
fn prepared_fft_matches_direct_oracles_for_every_supported_rank() {
    let Some(device) = device_or_skip() else {
        return;
    };
    verify_shape(&device, [8]);
    verify_shape(&device, [2, 4]);
    verify_shape(&device, [2, 2, 4]);
}

fn assert_fused_plan_shape<const R: usize>(
    device: &WgpuDevice,
    shape: [usize; R],
    active_axes: usize,
) {
    let elements = shape.into_iter().product();
    let real = device
        .upload(&vec![0.0_f32; elements])
        .expect("invariant: fused-plan real upload");
    let imaginary = device
        .upload(&vec![0.0_f32; elements])
        .expect("invariant: fused-plan imaginary upload");
    let layout = Layout::c_contiguous(shape).expect("invariant: dense fused-plan layout");
    let prepared = prepare(
        &WgpuFftOps,
        device,
        &real,
        &imaginary,
        &layout,
        FftDirection::Forward,
    );
    assert!(prepared.plan.workspace.is_none());
    assert_eq!(prepared.plan.fused_twiddle.is_some(), active_axes > 0);
    assert_eq!(prepared.plan.commands.len(), active_axes);
}

#[test]
fn fused_fft_skips_singleton_axes_and_allocates_no_volume_workspace() {
    let Some(device) = device_or_skip() else {
        return;
    };
    for shape in [[8, 1, 1], [1, 8, 1], [1, 1, 8]] {
        verify_shape(&device, shape);
        assert_fused_plan_shape(&device, shape, 1);
    }
    verify_shape(&device, [1, 1, 1]);
    assert_fused_plan_shape(&device, [1, 1, 1], 0);
    assert_fused_plan_shape(&device, [8, 4, 2], 3);
}

#[test]
fn prepared_bluestein_fft_matches_direct_oracles() {
    let Some(device) = device_or_skip() else {
        return;
    };
    verify_shape(&device, [3]);
    verify_shape(&device, [5]);
    verify_shape(&device, [2, 3]);
    verify_shape(&device, [2, 2, 3]);
}

#[test]
fn large_bluestein_impulse_preserves_range_reduced_phase() {
    let Some(device) = device_or_skip() else {
        return;
    };
    const N: usize = 262_147;
    const POSITION: usize = 250_000;
    let mut real_host = vec![0.0_f32; N];
    real_host[POSITION] = 1.0;
    let imaginary_host = vec![0.0_f32; N];
    let real = device.upload(&real_host).expect("invariant: real upload");
    let imaginary = device
        .upload(&imaginary_host)
        .expect("invariant: imaginary upload");
    let layout = Layout::c_contiguous([N]).expect("invariant: dense test layout");
    let ops = WgpuFftOps;
    let forward = prepare(
        &ops,
        &device,
        &real,
        &imaginary,
        &layout,
        FftDirection::Forward,
    );
    ops.dispatch_fft(&device, &forward)
        .expect("invariant: large Bluestein dispatch");

    let actual_real = download(&device, &real);
    let actual_imaginary = download(&device, &imaginary);
    let convolution_len = (2 * N - 1).next_power_of_two();
    let operation_depth = 2 * convolution_len.trailing_zeros() + 3;
    let tolerance = 512.0 * f64::from(f32::EPSILON) * f64::from(operation_depth);
    for output in [0, 1, N / 4, POSITION, N - 1] {
        let phase_index = (u64::try_from(POSITION).expect("test position fits u64")
            * u64::try_from(output).expect("test output fits u64"))
            % u64::try_from(N).expect("test length fits u64");
        let angle = -core::f64::consts::TAU * phase_index as f64 / f64::from(exact_u32(N));
        assert!(
            (f64::from(actual_real[output]) - angle.cos()).abs() <= tolerance,
            "real spectrum bin {output}: actual={} expected={} exceeds the depth-derived bound {tolerance}",
            actual_real[output],
            angle.cos(),
        );
        assert!(
            (f64::from(actual_imaginary[output]) - angle.sin()).abs() <= tolerance,
            "imaginary spectrum bin {output}: actual={} expected={} exceeds the depth-derived bound {tolerance}",
            actual_imaginary[output],
            angle.sin(),
        );
    }
}

#[test]
fn prepared_fft_rejects_cross_device_dispatch() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let other = WgpuDevice::try_default("hephaestus-fft-cross-device-test")
        .expect("invariant: a second logical WGPU device can be acquired");
    let real = device.upload(&[1.0, 2.0]).expect("invariant: upload");
    let imaginary = device.upload(&[0.0, 0.0]).expect("invariant: upload");
    let layout = Layout::c_contiguous([2]).expect("invariant: dense layout");
    let ops = WgpuFftOps;
    let prepared = prepare(
        &ops,
        &device,
        &real,
        &imaginary,
        &layout,
        FftDirection::Forward,
    );

    let expected = "kernel dispatch failed: prepared WGPU FFT belongs to a different device";
    assert_eq!(
        ops.dispatch_fft(&other, &prepared)
            .expect_err("cross-device dispatch must fail")
            .to_string(),
        expected
    );

    let mut stream = other.stream().expect("invariant: command stream");
    assert_eq!(
        stream
            .encode_grouped_sequence("hephaestus-fft-cross-device-pass", |sequence| {
                prepared.encode_in_sequence(sequence)
            })
            .expect_err("cross-device in-pass encoding must fail")
            .to_string(),
        expected
    );
}

#[test]
fn prepared_fft_owns_operands_and_encodes_in_existing_pass() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let shape = [2, 2, 2];
    let real_host = [0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0];
    let imaginary_host = [0.0; 8];
    let expected = direct_forward(shape, &real_host, &imaginary_host);
    let real = device.upload(&real_host).expect("invariant: real upload");
    let imaginary = device
        .upload(&imaginary_host)
        .expect("invariant: imaginary upload");
    let retained_real = real.clone();
    let retained_imaginary = imaginary.clone();
    let layout = Layout::c_contiguous(shape).expect("invariant: dense layout");
    let ops = WgpuFftOps;
    let prepared = prepare(
        &ops,
        &device,
        &real,
        &imaginary,
        &layout,
        FftDirection::Forward,
    );

    drop(real);
    drop(imaginary);

    let mut stream = device.stream().expect("invariant: command stream");
    stream
        .encode_grouped_sequence("hephaestus-fft-consumer-pass", |sequence| {
            sequence
                .raw_pass_mut()
                .insert_debug_marker("consumer-before-fft");
            prepared.encode_in_sequence(sequence)?;
            sequence
                .raw_pass_mut()
                .insert_debug_marker("consumer-after-fft");
            Ok(())
        })
        .expect("invariant: pass device owns the prepared FFT");
    stream
        .submit_with_timeout(std::time::Duration::from_secs(10))
        .expect("invariant: consumer pass submission completes");

    assert_complex_close(
        &download(&device, &retained_real),
        &download(&device, &retained_imaginary),
        &expected,
        shape.into_iter().product(),
    );
}
