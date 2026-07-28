//! Provider benchmark: CUDA execution against the canonical Leto CPU path.

use std::hint::black_box;
use std::time::{Duration, Instant};

use hephaestus_core::{BlockWidth, ComputeDevice};
use hephaestus_cuda::{
    AddOp, CudaDevice, StridedOperand, binary_elementwise_into, dot, matmul_into, norm_l1, norm_l2,
    norm_max, sum_axis_into,
};

const ELEMENTWISE_LEN: usize = 1 << 20;
const MATRIX_DIMENSION: usize = 256;
const ITERATIONS: usize = 50;

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
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("Skipping CUDA benchmark: {error}");
            return;
        }
    };

    benchmark_elementwise(&device);
    benchmark_reduction(&device);
    benchmark_map_reductions(&device);
    benchmark_matmul(&device);
}

fn benchmark_elementwise(device: &CudaDevice) {
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
    device.synchronize().unwrap();
    println!(
        "CUDA add: {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );
}

fn benchmark_reduction(device: &CudaDevice) {
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
    device.synchronize().unwrap();
    println!(
        "CUDA sum-axis: {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );
}

fn benchmark_map_reductions(device: &CudaDevice) {
    let host: Vec<f32> = (0..ELEMENTWISE_LEN)
        .map(|index| (index % 17) as f32 - 8.0)
        .collect();
    let rhs_host: Vec<f32> = (0..ELEMENTWISE_LEN)
        .map(|index| (index % 11) as f32 - 5.0)
        .collect();
    let input = device.upload(&host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = leto::Layout::c_contiguous([ELEMENTWISE_LEN]).unwrap();
    let operand = StridedOperand {
        buffer: &input,
        layout: &layout,
    };
    let rhs_operand = StridedOperand {
        buffer: &rhs,
        layout: &layout,
    };

    let expected_dot: f32 = host.iter().zip(&rhs_host).map(|(&a, &b)| a * b).sum();
    let expected_l1: f32 = host.iter().map(|value| value.abs()).sum();
    let expected_l2 = host.iter().map(|value| value * value).sum::<f32>().sqrt();
    let expected_max = host
        .iter()
        .map(|value| value.abs())
        .fold(f32::NEG_INFINITY, f32::max);

    let dot_output = dot(device, operand, rhs_operand).unwrap();
    let l1_output = norm_l1(device, operand).unwrap();
    let l2_output = norm_l2(device, operand).unwrap();
    let max_output = norm_max(device, operand).unwrap();
    device.synchronize().unwrap();

    let mut got = [0.0_f32; 1];
    device.download(&dot_output, &mut got).unwrap();
    let dot_tolerance = expected_dot.abs().max(1.0) * 1.0e-5;
    assert!((got[0] - expected_dot).abs() <= dot_tolerance);
    device.download(&l1_output, &mut got).unwrap();
    assert!((got[0] - expected_l1).abs() <= expected_l1 * 1.0e-5);
    device.download(&l2_output, &mut got).unwrap();
    assert!((got[0] - expected_l2).abs() <= expected_l2 * 1.0e-5);
    device.download(&max_output, &mut got).unwrap();
    assert_eq!(got[0], expected_max);

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(dot(device, black_box(operand), black_box(rhs_operand)).unwrap());
    }
    device.synchronize().unwrap();
    println!(
        "CUDA dot fused map-reduction: {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(norm_l1(device, black_box(operand)).unwrap());
    }
    device.synchronize().unwrap();
    println!(
        "CUDA norm_l1 fused map-reduction: {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(norm_l2(device, black_box(operand)).unwrap());
    }
    device.synchronize().unwrap();
    println!(
        "CUDA norm_l2 fused map-reduction: {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(norm_max(device, black_box(operand)).unwrap());
    }
    device.synchronize().unwrap();
    println!(
        "CUDA norm_max fused map-reduction: {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );
}

fn benchmark_matmul(device: &CudaDevice) {
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
    device.synchronize().unwrap();
    println!(
        "CUDA matmul: {} ns/iter",
        per_iteration(start.elapsed()).as_nanos()
    );
}
