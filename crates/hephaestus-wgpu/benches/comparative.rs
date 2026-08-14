#![expect(
    clippy::unwrap_used,
    reason = "ratchet HEPH-UNWRAP-1: pre-existing debt"
)]

//! Provider benchmark: WGPU execution against the canonical Leto CPU path.
//!
//! The benchmark keeps the reference implementation inside Atlas. It measures
//! real device dispatch and Leto operations over the same inputs; obsolete
//! external CPU baselines are intentionally absent from this provider.

use std::hint::black_box;
use std::time::{Duration, Instant};

use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer};
use hephaestus_wgpu::{
    AddOp, StridedOperand, SumOp, WgpuDevice, binary_elementwise_into, matmul_into,
    prepare_reduction, prepare_sum_axis_into, submit_prepared_axis_reduction_batch,
    submit_prepared_mixed_reduction_batch, submit_prepared_reduction_batch, sum_axis_into,
};

const ELEMENTWISE_LEN: usize = 1 << 20;
const MATRIX_DIMENSION: usize = 256;
const ITERATIONS: usize = 50;
const AXIS_BATCH_REDUCTIONS: usize = 8;
const SCALAR_BATCH_REDUCTIONS: usize = 8;
const SCALAR_REDUCTION_LEN: usize = 1 << 20;
const NOOP_BATCH_REDUCTIONS: usize = 8;
const DIRECT_NOOP_ITERATIONS: usize = 100_000;

fn per_iteration(elapsed: Duration) -> Duration {
    elapsed / u32::try_from(ITERATIONS).expect("invariant: benchmark iterations fit u32")
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "provider mismatch at {index}: got {actual}, expected {expected}"
        );
    }
}

fn main() {
    let device = match WgpuDevice::try_default("hephaestus-wgpu-comparative-bench") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("Skipping WGPU benchmark: {error}");
            return;
        }
    };

    benchmark_elementwise(&device);
    benchmark_reduction(&device);
    benchmark_matmul(&device);
}

fn benchmark_elementwise(device: &WgpuDevice) {
    let lhs_host: Vec<f32> = (0..ELEMENTWISE_LEN)
        .map(|index| (index as f32 * 0.731 + 1.0) * 1.0e-7)
        .collect();
    let rhs_host: Vec<f32> = (0..ELEMENTWISE_LEN)
        .map(|index| (index as f32 * 0.317 + 2.0) * 1.0e-7)
        .collect();
    let lhs = leto::Array::from_shape_vec([ELEMENTWISE_LEN], lhs_host.clone()).unwrap();
    let rhs = leto::Array::from_shape_vec([ELEMENTWISE_LEN], rhs_host.clone()).unwrap();
    let mut expected = leto::Array::zeros([ELEMENTWISE_LEN]);
    leto_ops::add(&lhs.view(), &rhs.view(), &mut expected.view_mut()).unwrap();
    let expected = leto::Storage::as_slice(expected.storage());

    let lhs_gpu = device.upload(&lhs_host).unwrap();
    let rhs_gpu = device.upload(&rhs_host).unwrap();
    let output_gpu = device.alloc_zeroed::<f32>(ELEMENTWISE_LEN).unwrap();
    binary_elementwise_into::<AddOp, f32>(
        device,
        &lhs_gpu,
        &rhs_gpu,
        &output_gpu,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut actual = vec![0.0; ELEMENTWISE_LEN];
    device.download(&output_gpu, &mut actual).unwrap();
    assert_close(&actual, expected, 1.0e-5);

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let mut output = leto::Array::zeros([ELEMENTWISE_LEN]);
        leto_ops::add(
            black_box(&lhs.view()),
            black_box(&rhs.view()),
            black_box(&mut output.view_mut()),
        )
        .unwrap();
        black_box(output);
    }
    println!(
        "Leto add: {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        binary_elementwise_into::<AddOp, f32>(
            device,
            black_box(&lhs_gpu),
            black_box(&rhs_gpu),
            black_box(&output_gpu),
            BlockWidth::DEFAULT,
        )
        .unwrap();
    }
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    println!(
        "WGPU add: {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );
}

fn benchmark_reduction(device: &WgpuDevice) {
    let shape = [MATRIX_DIMENSION, MATRIX_DIMENSION];
    let host: Vec<f32> = (0..MATRIX_DIMENSION * MATRIX_DIMENSION)
        .map(|index| (index % 17) as f32 - 8.0)
        .collect();
    let input = leto::Array::from_shape_vec(shape, host.clone()).unwrap();
    let expected = leto_ops::sum_axis(&input.view(), 0).unwrap();
    let expected = leto::Storage::as_slice(expected.storage());
    let input_gpu = device.upload(&host).unwrap();
    let output_gpu = device.alloc_zeroed::<f32>(MATRIX_DIMENSION).unwrap();
    let input_layout = leto::Layout::c_contiguous(shape).unwrap();
    let output_layout = leto::Layout::c_contiguous([1, MATRIX_DIMENSION]).unwrap();
    sum_axis_into(
        device,
        StridedOperand {
            buffer: &input_gpu,
            layout: &input_layout,
        },
        0,
        StridedOperand {
            buffer: &output_gpu,
            layout: &output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut actual = vec![0.0; MATRIX_DIMENSION];
    device.download(&output_gpu, &mut actual).unwrap();
    assert_close(&actual, expected, 1.0e-5);

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let mut output = leto::Array::zeros([1, MATRIX_DIMENSION]);
        leto_ops::sum_axis_into(&input.view(), 0, &mut output.view_mut()).unwrap();
        black_box(output);
    }
    println!(
        "Leto sum-axis: {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        sum_axis_into(
            device,
            black_box(StridedOperand {
                buffer: &input_gpu,
                layout: &input_layout,
            }),
            0,
            black_box(StridedOperand {
                buffer: &output_gpu,
                layout: &output_layout,
            }),
            BlockWidth::DEFAULT,
        )
        .unwrap();
    }
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    println!(
        "WGPU sum-axis: {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );

    let batch_outputs: Vec<_> = (0..AXIS_BATCH_REDUCTIONS)
        .map(|_| device.alloc_zeroed::<f32>(MATRIX_DIMENSION).unwrap())
        .collect();
    let prepared_batch: Vec<_> = batch_outputs
        .iter()
        .map(|output| {
            prepare_sum_axis_into(
                device,
                StridedOperand {
                    buffer: &input_gpu,
                    layout: &input_layout,
                },
                0,
                StridedOperand {
                    buffer: output,
                    layout: &output_layout,
                },
                BlockWidth::DEFAULT,
            )
            .unwrap()
        })
        .collect();
    let prepared_batch: Vec<_> = prepared_batch.iter().collect();
    submit_prepared_axis_reduction_batch(device, &prepared_batch).unwrap();
    for output in &batch_outputs {
        device.download(output, &mut actual).unwrap();
        assert_close(&actual, expected, 1.0e-5);
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        submit_prepared_axis_reduction_batch(device, black_box(&prepared_batch)).unwrap();
    }
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    println!(
        "WGPU prepared sum-axis batch ({AXIS_BATCH_REDUCTIONS}): {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );

    let scalar_host: Vec<u32> = (0..SCALAR_REDUCTION_LEN)
        .map(|index| u32::try_from(index % 17).unwrap())
        .collect();
    let scalar_expected: u32 = scalar_host.iter().sum();
    let scalar_input = device.upload(&scalar_host).unwrap();
    let scalar_batch: Vec<_> = (0..SCALAR_BATCH_REDUCTIONS)
        .map(|_| prepare_reduction::<SumOp, u32>(device, &scalar_input).unwrap())
        .collect();
    let scalar_batch: Vec<_> = scalar_batch.iter().collect();
    submit_prepared_reduction_batch(device, &scalar_batch).unwrap();
    for reduction in &scalar_batch {
        let mut actual = [0u32; 1];
        device.download(reduction.output(), &mut actual).unwrap();
        assert_eq!(actual[0], scalar_expected);
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        submit_prepared_reduction_batch(device, black_box(&scalar_batch)).unwrap();
    }
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    println!(
        "WGPU prepared scalar-sum batch ({SCALAR_BATCH_REDUCTIONS}): {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );

    submit_prepared_mixed_reduction_batch(device, &scalar_batch, &prepared_batch).unwrap();
    for reduction in &scalar_batch {
        let mut scalar_actual = [0u32; 1];
        device
            .download(reduction.output(), &mut scalar_actual)
            .unwrap();
        assert_eq!(scalar_actual, [scalar_expected]);
    }
    for output in &batch_outputs {
        device.download(output, &mut actual).unwrap();
        assert_close(&actual, expected, 1.0e-5);
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        submit_prepared_reduction_batch(device, black_box(&scalar_batch)).unwrap();
        submit_prepared_axis_reduction_batch(device, black_box(&prepared_batch)).unwrap();
    }
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    println!(
        "WGPU separate scalar + axis batches ({SCALAR_BATCH_REDUCTIONS} + {AXIS_BATCH_REDUCTIONS}): {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        submit_prepared_mixed_reduction_batch(
            device,
            black_box(&scalar_batch),
            black_box(&prepared_batch),
        )
        .unwrap();
    }
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    println!(
        "WGPU mixed scalar + axis batch ({SCALAR_BATCH_REDUCTIONS} + {AXIS_BATCH_REDUCTIONS}): {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );

    let empty_scalar_input = device.upload::<u32>(&[]).unwrap();
    let empty_scalar_batch: Vec<_> = (0..NOOP_BATCH_REDUCTIONS)
        .map(|_| prepare_reduction::<SumOp, u32>(device, &empty_scalar_input).unwrap())
        .collect();
    let empty_scalar_batch: Vec<_> = empty_scalar_batch.iter().collect();
    let empty_axis_input = device.upload::<f32>(&[]).unwrap();
    let empty_axis_output = device.alloc_zeroed::<f32>(0).unwrap();
    let empty_axis_input_layout = leto::Layout::c_contiguous([0, 3]).unwrap();
    let empty_axis_output_layout = leto::Layout::c_contiguous([0, 1]).unwrap();
    let empty_axis_batch: Vec<_> = (0..NOOP_BATCH_REDUCTIONS)
        .map(|_| {
            prepare_sum_axis_into(
                device,
                StridedOperand {
                    buffer: &empty_axis_input,
                    layout: &empty_axis_input_layout,
                },
                1,
                StridedOperand {
                    buffer: &empty_axis_output,
                    layout: &empty_axis_output_layout,
                },
                BlockWidth::DEFAULT,
            )
            .unwrap()
        })
        .collect();
    let empty_axis_batch: Vec<_> = empty_axis_batch.iter().collect();
    submit_prepared_reduction_batch(device, &empty_scalar_batch).unwrap();
    submit_prepared_axis_reduction_batch(device, &empty_axis_batch).unwrap();
    for reduction in &empty_scalar_batch {
        let mut actual = [u32::MAX; 1];
        device.download(reduction.output(), &mut actual).unwrap();
        assert_eq!(actual, [0]);
    }
    assert_eq!(empty_axis_output.len(), 0);

    empty_scalar_batch[0].dispatch(device).unwrap();
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let start = Instant::now();
    for _ in 0..DIRECT_NOOP_ITERATIONS {
        black_box(empty_scalar_batch[0]).dispatch(device).unwrap();
    }
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    println!(
        "WGPU direct empty prepared scalar reduction ({}-byte inline host plan): {:.3} ns/iter",
        std::mem::size_of_val(empty_scalar_batch[0]),
        start.elapsed().as_secs_f64() * 1.0e9
            / f64::from(
                u32::try_from(DIRECT_NOOP_ITERATIONS)
                    .expect("invariant: direct no-op iterations fit u32"),
            )
    );

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        submit_prepared_reduction_batch(device, black_box(&empty_scalar_batch)).unwrap();
        submit_prepared_axis_reduction_batch(device, black_box(&empty_axis_batch)).unwrap();
    }
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    println!(
        "WGPU prepared no-op batches ({NOOP_BATCH_REDUCTIONS} scalar + axis): {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );
}

fn benchmark_matmul(device: &WgpuDevice) {
    let n = 64;
    let lhs_host: Vec<f32> = (0..n * n).map(|index| (index % 11) as f32 - 5.0).collect();
    let rhs_host: Vec<f32> = (0..n * n).map(|index| (index % 7) as f32 - 3.0).collect();
    let lhs = leto::Array::from_shape_vec([n, n], lhs_host.clone()).unwrap();
    let rhs = leto::Array::from_shape_vec([n, n], rhs_host.clone()).unwrap();
    let mut expected = leto::Array::zeros([n, n]);
    leto_ops::matmul(&lhs.view(), &rhs.view(), &mut expected.view_mut()).unwrap();
    let expected = leto::Storage::as_slice(expected.storage());

    let lhs_gpu = device.upload(&lhs_host).unwrap();
    let rhs_gpu = device.upload(&rhs_host).unwrap();
    let output_gpu = device.alloc_zeroed::<f32>(n * n).unwrap();
    let layout = leto::Layout::c_contiguous([n, n]).unwrap();
    matmul_into(
        device,
        StridedOperand {
            buffer: &lhs_gpu,
            layout: &layout,
        },
        StridedOperand {
            buffer: &rhs_gpu,
            layout: &layout,
        },
        StridedOperand {
            buffer: &output_gpu,
            layout: &layout,
        },
    )
    .unwrap();
    let mut actual = vec![0.0; n * n];
    device.download(&output_gpu, &mut actual).unwrap();
    assert_close(&actual, expected, 1.0e-4);

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let mut output = leto::Array::zeros([n, n]);
        leto_ops::matmul(
            black_box(&lhs.view()),
            black_box(&rhs.view()),
            black_box(&mut output.view_mut()),
        )
        .unwrap();
        black_box(output);
    }
    println!(
        "Leto matmul: {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        matmul_into(
            device,
            black_box(StridedOperand {
                buffer: &lhs_gpu,
                layout: &layout,
            }),
            black_box(StridedOperand {
                buffer: &rhs_gpu,
                layout: &layout,
            }),
            black_box(StridedOperand {
                buffer: &output_gpu,
                layout: &layout,
            }),
        )
        .unwrap();
    }
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    println!(
        "WGPU matmul: {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );
}
