use std::{hint::black_box, time::Duration};

use bytemuck::Pod;
use criterion::{BenchmarkId, Criterion, Throughput};
use eunomia::F16;
use hephaestus_core::{
    DeviceFeature, DevicePreference, FftDirection, FftOperands, FftOps, KernelDevice, StridedView,
};
use hephaestus_wgpu::{
    ComputeDevice, DeviceBuffer, WgpuBuffer, WgpuDevice, WgpuFftOps, WgpuFftScalar,
};
use leto::Layout;

const GPU_WAIT_TIMEOUT: Duration = Duration::from_secs(10);

trait BenchScalar: WgpuFftScalar + Pod + Copy {
    const LABEL: &'static str;
    const UNIT_ROUNDOFF: f64;

    fn from_input(value: f32) -> Self;
    fn to_output(self) -> f32;
}

impl BenchScalar for f32 {
    const LABEL: &'static str = "binary32";
    const UNIT_ROUNDOFF: f64 = f32::EPSILON as f64 / 2.0;

    fn from_input(value: f32) -> Self {
        value
    }

    fn to_output(self) -> f32 {
        self
    }
}

impl BenchScalar for F16 {
    const LABEL: &'static str = "binary16";
    const UNIT_ROUNDOFF: f64 = 1.0 / 2048.0;

    fn from_input(value: f32) -> Self {
        Self::from_f32(value)
    }

    fn to_output(self) -> f32 {
        self.to_f32()
    }
}

fn operands<'a, T, const R: usize>(
    real: &'a WgpuBuffer<T>,
    imaginary: &'a WgpuBuffer<T>,
    layout: &'a Layout<R>,
) -> FftOperands<'a, WgpuBuffer<T>, R> {
    FftOperands {
        real: StridedView::new(real, layout),
        imaginary: StridedView::new(imaginary, layout),
    }
}

fn download<T: BenchScalar>(device: &WgpuDevice, buffer: &WgpuBuffer<T>) -> Vec<T> {
    let mut host = vec![T::from_input(0.0); buffer.len()];
    device
        .download_with_timeout(buffer, &mut host, GPU_WAIT_TIMEOUT)
        .expect("bounded scalar FFT validation download");
    host
}

fn mode_index<const R: usize>(shape: [usize; R], angular_frequency: f64) -> usize {
    let mut linear = 0usize;
    for axis in 0..R {
        let stride = shape[(axis + 1)..].iter().product::<usize>();
        let cycles = angular_frequency.abs()
            * f64::from(super::exact_u32(stride))
            * f64::from(super::exact_u32(shape[axis]))
            / core::f64::consts::TAU;
        let positive = cycles.round() as usize % shape[axis];
        let coordinate = if angular_frequency.is_sign_negative() && positive != 0 {
            shape[axis] - positive
        } else {
            positive
        };
        linear = linear * shape[axis] + coordinate;
    }
    linear
}

fn validate_forward<T: BenchScalar, const R: usize>(
    device: &WgpuDevice,
    real: &WgpuBuffer<T>,
    imaginary: &WgpuBuffer<T>,
    input_real: &[T],
    input_imaginary: &[T],
    shape: [usize; R],
) {
    let actual_real = download(device, real);
    let actual_imaginary = download(device, imaginary);
    let expected_real = input_real
        .iter()
        .map(|value| value.to_output())
        .collect::<Vec<_>>();
    let expected_imaginary = input_imaginary
        .iter()
        .map(|value| value.to_output())
        .collect::<Vec<_>>();
    let stages = shape
        .into_iter()
        .filter(|&length| length > 1)
        .map(usize::ilog2)
        .sum::<u32>();
    let accumulated = f64::from(1 + 5 * stages) * T::UNIT_ROUNDOFF;
    assert!(
        accumulated < 1.0,
        "forward benchmark error model requires sites * unit roundoff < 1"
    );
    let gamma = accumulated / (1.0 - accumulated);
    let component_bound = gamma
        * expected_real
            .iter()
            .zip(&expected_imaginary)
            .map(|(&real, &imaginary)| f64::from(real).hypot(f64::from(imaginary)))
            .sum::<f64>();
    let elements = expected_real.len();
    let mut samples = vec![
        0,
        mode_index(shape, 0.017),
        mode_index(shape, -0.017),
        mode_index(shape, 0.031),
        mode_index(shape, -0.031),
    ];
    samples.sort_unstable();
    samples.dedup();
    let mut rejects_identity = false;
    for index in samples {
        let expected = super::direct_bin(shape, &expected_real, &expected_imaginary, index);
        let error = (f64::from(actual_real[index].to_output()) - expected[0])
            .hypot(f64::from(actual_imaginary[index].to_output()) - expected[1]);
        assert!(
            error <= component_bound,
            "{} forward DFT error {error:.3e} exceeds derived bound {component_bound:.3e} at {index}/{elements}",
            T::LABEL,
        );
        let identity_error = (f64::from(expected_real[index]) - expected[0])
            .hypot(f64::from(expected_imaginary[index]) - expected[1]);
        rejects_identity |= identity_error > component_bound;
    }
    assert!(
        rejects_identity,
        "{} sampled forward oracle does not discriminate an identity transform",
        T::LABEL
    );
}

fn validate_roundtrip<T: BenchScalar, const R: usize>(
    device: &WgpuDevice,
    real: &WgpuBuffer<T>,
    imaginary: &WgpuBuffer<T>,
    expected_real: &[f32],
    expected_imaginary: &[f32],
    shape: [usize; R],
) {
    let stages = shape
        .into_iter()
        .filter(|&length| length > 1)
        .map(usize::ilog2)
        .sum::<u32>();
    // Input quantization, five rounding sites per radix stage and component
    // direction, plus one inverse scale per active axis.
    let sites = 1 + 10 * stages + shape.into_iter().filter(|&length| length > 1).count() as u32;
    let accumulated = f64::from(sites) * T::UNIT_ROUNDOFF;
    let gamma = accumulated / (1.0 - accumulated);
    let actual_real = download(device, real);
    let actual_imaginary = download(device, imaginary);
    let (error_squared, expected_squared) = actual_real
        .into_iter()
        .zip(actual_imaginary)
        .zip(expected_real.iter().zip(expected_imaginary))
        .fold(
            (0.0, 0.0),
            |(error_squared, expected_squared),
             ((real, imaginary), (&expected_real, &expected_imaginary))| {
                let real_error = f64::from(real.to_output()) - f64::from(expected_real);
                let imaginary_error =
                    f64::from(imaginary.to_output()) - f64::from(expected_imaginary);
                (
                    real_error.mul_add(
                        real_error,
                        imaginary_error.mul_add(imaginary_error, error_squared),
                    ),
                    f64::from(expected_real).mul_add(
                        f64::from(expected_real),
                        f64::from(expected_imaginary)
                            .mul_add(f64::from(expected_imaginary), expected_squared),
                    ),
                )
            },
        );
    assert!(
        expected_squared > 0.0,
        "benchmark oracle requires nonzero input energy"
    );
    let relative_error = (error_squared / expected_squared).sqrt();
    assert!(
        relative_error <= gamma,
        "{} round-trip relative L2 error {relative_error:.3e} exceeds derived bound {gamma:.3e}",
        T::LABEL
    );
}

fn bench_shape<T: BenchScalar, const R: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    device: &WgpuDevice,
    shape: [usize; R],
) {
    let elements = shape.into_iter().product();
    // The forward transform is intentionally unnormalized. Scaling by
    // 1/sqrt(N) keeps the transform's intermediate values within binary16's
    // finite range without changing the dispatch or memory workload.
    let input_scale = 1.0 / (elements as f32).sqrt();
    let real_host = (0..elements)
        .map(|index| input_scale * (0.017 * index as f32).sin())
        .collect::<Vec<_>>();
    let imaginary_host = (0..elements)
        .map(|index| 0.25 * input_scale * (0.031 * index as f32).cos())
        .collect::<Vec<_>>();
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
    let forward = ops
        .prepare_fft(
            device,
            operands(&real, &imaginary, &layout),
            FftDirection::Forward,
        )
        .expect("scalar FFT forward preparation");
    let inverse = ops
        .prepare_fft(
            device,
            operands(&real, &imaginary, &layout),
            FftDirection::Inverse,
        )
        .expect("scalar FFT inverse preparation");
    ops.dispatch_fft(device, &forward)
        .expect("scalar FFT forward validation");
    validate_forward(
        device,
        &real,
        &imaginary,
        &real_storage,
        &imaginary_storage,
        shape,
    );
    ops.dispatch_fft(device, &inverse)
        .expect("scalar FFT inverse validation");
    validate_roundtrip(
        device,
        &real,
        &imaginary,
        &real_host,
        &imaginary_host,
        shape,
    );

    group.throughput(Throughput::Elements(
        u64::try_from(elements)
            .expect("benchmark element count fits u64")
            .checked_mul(2)
            .expect("paired benchmark element count fits u64"),
    ));
    group.bench_with_input(BenchmarkId::new(T::LABEL, elements), &(), |b, ()| {
        b.iter(|| {
            let mut stream = device.stream().expect("scalar FFT benchmark stream");
            ops.encode_fft(device, &forward, &mut stream)
                .expect("scalar FFT forward encoding");
            ops.encode_fft(device, &inverse, &mut stream)
                .expect("scalar FFT inverse encoding");
            stream
                .submit_with_timeout(GPU_WAIT_TIMEOUT)
                .expect("bounded scalar FFT benchmark submission");
            black_box((real.raw(), imaginary.raw()))
        });
    });
}

pub(super) fn bench_scalar_comparison(c: &mut Criterion) {
    let device = match WgpuDevice::try_with_device_preference_and_required_device_features(
        "hephaestus-prepared-fft-scalar-bench",
        DevicePreference::HighPerformance,
        &[DeviceFeature::ShaderF16],
    ) {
        Ok(device) => device,
        Err(error) => {
            if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_some() {
                panic!(
                    "WGPU ShaderF16 adapter required, but benchmark acquisition failed: {error}"
                );
            }
            eprintln!("skipping WGPU FFT scalar benchmark: {error}");
            return;
        }
    };
    if let Some(adapter) = device.adapter_info() {
        eprintln!(
            "prepared FFT scalar adapter: {} ({:?}, {:?}), driver {} {}",
            adapter.name, adapter.backend, adapter.device_type, adapter.driver, adapter.driver_info
        );
    }
    let mut group = c.benchmark_group("prepared_fft_scalar");
    bench_shape::<f32, 1>(&mut group, &device, [65_536]);
    bench_shape::<F16, 1>(&mut group, &device, [65_536]);
    bench_shape::<f32, 3>(&mut group, &device, [64, 64, 64]);
    bench_shape::<F16, 3>(&mut group, &device, [64, 64, 64]);
    group.finish();
}
