//! Prepared WGPU FFT throughput over one-, two-, and three-dimensional shapes.
//!
//! Every timed iteration encodes a forward and inverse transform into one
//! command stream, submits once, and waits for device completion. Preparation,
//! host transfer, and validation stay outside the timed region. The paired
//! transform preserves the input distribution across Criterion iterations.

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hephaestus_core::{DeviceBuffer, KernelDevice, StridedView};
use hephaestus_wgpu::{
    ComputeDevice, FftDirection, FftOperands, FftOps, WgpuBuffer, WgpuDevice, WgpuFftOps,
    WgpuPreparedFft,
};
use leto::Layout;

const GPU_WAIT_TIMEOUT: Duration = Duration::from_secs(10);
// A radix stage performs at most one complex multiply and a butterfly per
// component. Sixteen rounded real operations conservatively covers that work;
// Bluestein counts its two convolution FFTs and three pointwise passes below.
const ROUNDED_REAL_OPS_PER_STAGE: u32 = 16;

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

fn exact_u32(value: usize) -> u32 {
    u32::try_from(value).expect("benchmark dimension fits u32")
}

fn coordinate<const R: usize>(mut linear: usize, shape: [usize; R]) -> [usize; R] {
    let mut coordinate = [0; R];
    for axis in (0..R).rev() {
        coordinate[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    coordinate
}

fn direct_bin<const R: usize>(
    shape: [usize; R],
    real: &[f32],
    imaginary: &[f32],
    output_index: usize,
) -> [f64; 2] {
    let output = coordinate(output_index, shape);
    real.iter().zip(imaginary).enumerate().fold(
        [0.0, 0.0],
        |sum, (input_index, (&real, &imaginary))| {
            let input = coordinate(input_index, shape);
            let phase = (0..R).fold(0.0, |phase, axis| {
                phase
                    + f64::from(exact_u32(input[axis])) * f64::from(exact_u32(output[axis]))
                        / f64::from(exact_u32(shape[axis]))
            });
            let angle = -core::f64::consts::TAU * phase;
            let twiddle = [angle.cos(), angle.sin()];
            [
                sum[0] + f64::from(real).mul_add(twiddle[0], -(f64::from(imaginary) * twiddle[1])),
                sum[1] + f64::from(real).mul_add(twiddle[1], f64::from(imaginary) * twiddle[0]),
            ]
        },
    )
}

fn rounded_operations<const R: usize>(shape: [usize; R]) -> u32 {
    shape
        .into_iter()
        .map(|length| {
            if length.is_power_of_two() {
                exact_u32(length).ilog2()
            } else {
                let convolution = length
                    .checked_mul(2)
                    .and_then(|twice| twice.checked_sub(1))
                    .and_then(usize::checked_next_power_of_two)
                    .expect("benchmark Bluestein workspace fits usize");
                exact_u32(convolution)
                    .ilog2()
                    .saturating_mul(2)
                    .saturating_add(3)
            }
        })
        .sum::<u32>()
        .saturating_mul(ROUNDED_REAL_OPS_PER_STAGE)
}

fn gamma(rounded_operations: u32) -> f64 {
    let accumulated = f64::from(rounded_operations) * f64::from(f32::EPSILON);
    assert!(
        accumulated < 1.0,
        "benchmark error model requires k*epsilon < 1"
    );
    accumulated / (1.0 - accumulated)
}

fn download(device: &WgpuDevice, buffer: &WgpuBuffer<f32>) -> Vec<f32> {
    let mut host = vec![0.0; buffer.len()];
    device
        .download_with_timeout(buffer, &mut host, GPU_WAIT_TIMEOUT)
        .expect("invariant: bounded benchmark validation download succeeds");
    host
}

fn assert_forward_bins<const R: usize>(
    device: &WgpuDevice,
    actual_real: &WgpuBuffer<f32>,
    actual_imaginary: &WgpuBuffer<f32>,
    expected_real: &[f32],
    expected_imaginary: &[f32],
    shape: [usize; R],
) {
    let real = download(device, actual_real);
    let imaginary = download(device, actual_imaginary);
    let elements = real.len();
    let l1_norm = expected_real
        .iter()
        .zip(expected_imaginary)
        .map(|(&real, &imaginary)| f64::from(real).abs() + f64::from(imaginary).abs())
        .sum::<f64>();
    let tolerance = gamma(rounded_operations(shape)) * l1_norm.max(1.0);
    for index in [0, elements / 3, elements - 1] {
        let expected = direct_bin(shape, expected_real, expected_imaginary, index);
        assert!(
            (f64::from(real[index]) - expected[0]).abs() <= tolerance,
            "forward real[{index}]={}, expected {}, tolerance {tolerance}",
            real[index],
            expected[0]
        );
        assert!(
            (f64::from(imaginary[index]) - expected[1]).abs() <= tolerance,
            "forward imaginary[{index}]={}, expected {}, tolerance {tolerance}",
            imaginary[index],
            expected[1]
        );
    }
}

fn assert_round_trip<const R: usize>(
    device: &WgpuDevice,
    actual: &WgpuBuffer<f32>,
    expected: &[f32],
    shape: [usize; R],
) {
    let host = download(device, actual);
    let tolerance = gamma(rounded_operations(shape).saturating_mul(2));
    for (index, (&actual, &expected)) in host.iter().zip(expected).enumerate() {
        assert!(
            (f64::from(actual) - f64::from(expected)).abs()
                <= tolerance * f64::from(expected).abs().max(1.0),
            "round-trip[{index}]={actual}, expected {expected}"
        );
    }
}

struct PreparedValidation<'view, 'plan, const R: usize> {
    forward: &'view WgpuPreparedFft<'plan, R>,
    inverse: &'view WgpuPreparedFft<'plan, R>,
    real: &'view WgpuBuffer<f32>,
    imaginary: &'view WgpuBuffer<f32>,
    real_host: &'view [f32],
    imaginary_host: &'view [f32],
    shape: [usize; R],
}

fn validate_prepared<const R: usize>(
    device: &WgpuDevice,
    validation: PreparedValidation<'_, '_, R>,
) {
    let PreparedValidation {
        forward,
        inverse,
        real,
        imaginary,
        real_host,
        imaginary_host,
        shape,
    } = validation;
    let ops = WgpuFftOps;
    let mut stream = device.stream().expect("invariant: validation stream");
    ops.encode_fft(device, forward, &mut stream)
        .expect("invariant: validation forward encoding");
    stream
        .submit_with_timeout(GPU_WAIT_TIMEOUT)
        .expect("invariant: bounded validation forward submission");
    assert_forward_bins(device, real, imaginary, real_host, imaginary_host, shape);

    let mut stream = device.stream().expect("invariant: validation stream");
    ops.encode_fft(device, inverse, &mut stream)
        .expect("invariant: validation inverse encoding");
    stream
        .submit_with_timeout(GPU_WAIT_TIMEOUT)
        .expect("invariant: bounded validation inverse submission");
    assert_round_trip(device, real, real_host, shape);
    assert_round_trip(device, imaginary, imaginary_host, shape);
}

fn bench_shape<const R: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    device: &WgpuDevice,
    label: &str,
    shape: [usize; R],
) {
    let elements = shape.into_iter().product();
    let real_host = (0..elements)
        .map(|index| {
            f32::from(u16::try_from(index % 251).expect("bounded benchmark value")) / 251.0
        })
        .collect::<Vec<_>>();
    let imaginary_host = (0..elements)
        .map(|index| {
            f32::from(u16::try_from(index % 127).expect("bounded benchmark value")) / 127.0 - 0.5
        })
        .collect::<Vec<_>>();
    let real = device.upload(&real_host).expect("invariant: real upload");
    let imaginary = device
        .upload(&imaginary_host)
        .expect("invariant: imaginary upload");
    let layout = Layout::c_contiguous(shape).expect("invariant: dense benchmark layout");
    let ops = WgpuFftOps;
    let forward = ops
        .prepare_fft(
            device,
            operands(&real, &imaginary, &layout),
            FftDirection::Forward,
        )
        .expect("invariant: forward preparation");
    let inverse = ops
        .prepare_fft(
            device,
            operands(&real, &imaginary, &layout),
            FftDirection::Inverse,
        )
        .expect("invariant: inverse preparation");

    validate_prepared(
        device,
        PreparedValidation {
            forward: &forward,
            inverse: &inverse,
            real: &real,
            imaginary: &imaginary,
            real_host: &real_host,
            imaginary_host: &imaginary_host,
            shape,
        },
    );

    group.throughput(Throughput::Elements(
        u64::try_from(elements)
            .expect("benchmark element count fits u64")
            .saturating_mul(2),
    ));
    group.bench_with_input(BenchmarkId::new("forward_inverse", label), &(), |b, ()| {
        b.iter(|| {
            let mut stream = device.stream().expect("invariant: benchmark stream");
            ops.encode_fft(device, &forward, &mut stream)
                .expect("invariant: forward encoding");
            ops.encode_fft(device, &inverse, &mut stream)
                .expect("invariant: inverse encoding");
            stream
                .submit_with_timeout(GPU_WAIT_TIMEOUT)
                .expect("invariant: bounded benchmark submission");
            black_box((real.raw(), imaginary.raw()))
        });
    });
}

fn prepared_fft(c: &mut Criterion) {
    let device = match WgpuDevice::try_default("hephaestus-prepared-fft-bench") {
        Ok(device) => device,
        Err(error) => {
            if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_some() {
                panic!("WGPU adapter required, but benchmark acquisition failed: {error}");
            }
            eprintln!("skipping WGPU FFT benchmark: {error}");
            return;
        }
    };
    let mut group = c.benchmark_group("prepared_fft");
    bench_shape(&mut group, &device, "1d_1024", [1024]);
    bench_shape(&mut group, &device, "1d_1000_bluestein", [1000]);
    bench_shape(&mut group, &device, "1d_65536", [65_536]);
    bench_shape(&mut group, &device, "2d_256x256", [256, 256]);
    bench_shape(&mut group, &device, "3d_64x64x64", [64, 64, 64]);
    bench_shape(&mut group, &device, "3d_32x32x33_bluestein", [32, 32, 33]);
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = prepared_fft
}
criterion_main!(benches);
