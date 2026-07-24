//! Value-semantic contracts for the ROCm device substrate.
//!
//! The default build verifies the typed unavailable path. With `--features
//! rocm`, the same tests execute against HIP when an AMD device is present.
//! Hardware CI sets `HEPHAESTUS_ROCM_REQUIRE_DEVICE=1` so an unavailable device
//! fails that lane instead of being reported as device evidence.

use hephaestus_core::{
    AddOp, BlockWidth, ComputeDevice, ComputeDeviceCapabilities, DeviceBuffer, DeviceFeature,
    HephaestusError, MaxOp, MinOp, MulOp, NegOp, SumOp,
};
use hephaestus_rocm::{
    CumSumOp, Result, RocmDevice, ScanDirection, StridedOperand, batched_matmul,
    batched_matmul_into, binary_elementwise, binary_elementwise_into, cumprod, cumsum, matmul,
    matmul_into, max_axis, mean_axis, mean_axis_into, min_axis, reduction_with_width,
    scalar_elementwise, scan_axis, scan_axis_into, sum_axis, unary_elementwise,
};
use leto::Layout;

fn device(test: &str) -> Option<RocmDevice> {
    match RocmDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) => {
            if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_some() {
                panic!("ROCm device required for {test}, but acquisition failed: {error}");
            }
            eprintln!("skip {test}: ROCm device unavailable ({error})");
            None
        }
    }
}

fn assert_length_mismatch<T>(result: Result<T>, host_len: usize, device_len: usize) {
    match result {
        Err(HephaestusError::LengthMismatch {
            host_len: actual_host_len,
            device_len: actual_device_len,
        }) => {
            assert_eq!(actual_host_len, host_len);
            assert_eq!(actual_device_len, device_len);
        }
        Err(error) => panic!("expected length mismatch, got {error:?}"),
        Ok(_) => panic!("expected length mismatch {host_len}->{device_len}, got success"),
    }
}

#[test]
fn backend_name_is_stable() {
    #[cfg(not(feature = "rocm"))]
    assert_eq!(
        RocmDevice::try_default().unwrap_err().to_string(),
        "no compatible accelerator adapter available: ROCm backend unavailable: enable the `rocm` feature on Linux with ROCm installed"
    );

    #[cfg(feature = "rocm")]
    if let Some(device) = device("backend_name_is_stable") {
        assert_eq!(device.backend_name(), "rocm");
    }
}

#[test]
fn device_capabilities_and_topology_are_driver_backed() {
    let Some(device) = device("device_capabilities_and_topology_are_driver_backed") else {
        return;
    };

    let limits = device.device_limits();
    assert!(limits.max_buffer_size > 0);
    assert!(limits.max_compute_workgroup_size_x > 0);
    assert!(limits.max_compute_workgroup_size_y > 0);
    assert!(limits.max_compute_workgroup_size_z > 0);
    assert!(limits.max_compute_invocations_per_workgroup > 0);
    assert!(limits.max_compute_workgroup_storage_size > 0);
    assert_eq!(limits.max_storage_buffers_per_shader_stage, None);
    assert_eq!(
        limits.max_buffers_and_acceleration_structures_per_shader_stage,
        None
    );
    assert_eq!(limits.max_immediate_size, 0);

    assert!(!device.supports_device_feature(DeviceFeature::TimestampQuery));
    assert!(!device.supports_device_feature(DeviceFeature::ShaderF64));
    assert!(!device.supports_device_feature(DeviceFeature::ShaderF16));
    assert!(!device.supports_device_feature(DeviceFeature::ImmediateData));

    let topology = device
        .topology()
        .expect("an acquired ROCm device must have a topology snapshot");
    assert!(topology.compute_units() > 0);
    assert!(topology.warp_width() > 0);
    assert!(topology.max_threads_per_unit() > 0);
    assert!(topology.registers_per_unit() > 0);
    assert!(topology.shared_mem_per_unit_bytes() > 0);
    assert!(topology.memory_bytes() > 0);
    assert_eq!(topology.memory_tier(), themis::MemoryTier::Device);
    assert!(topology.max_resident_warps() > 0);
}

#[test]
fn upload_download_roundtrip_preserves_values() {
    let Some(device) = device("upload_download_roundtrip_preserves_values") else {
        return;
    };

    let host = [1.0_f32, 2.0, -3.5, 4.25, 0.0, 1024.5];
    let buffer = device.upload(&host).expect("HIP upload");
    assert_eq!(buffer.len(), host.len());
    assert_eq!(buffer.tier(), themis::MemoryTier::Device);

    let mut output = [0.0_f32; 6];
    device.download(&buffer, &mut output).expect("HIP download");
    device.synchronize().expect("HIP synchronization");
    assert_eq!(output, host);
}

#[test]
fn alloc_zeroed_produces_zero_values() {
    let Some(device) = device("alloc_zeroed_produces_zero_values") else {
        return;
    };

    let buffer = device.alloc_zeroed::<u32>(17).expect("HIP allocation");
    let mut output = [9_u32; 17];
    device.download(&buffer, &mut output).expect("HIP download");
    assert_eq!(output, [0_u32; 17]);
}

#[test]
fn placement_hints_match_hip_allocation_contract() {
    let Some(device) = device("placement_hints_match_hip_allocation_contract") else {
        return;
    };

    let host_visible = device
        .alloc_zeroed_with_hint::<u32>(
            4,
            themis::PlacementHint::Tier(themis::MemoryTier::HostPinned),
        )
        .expect("host-visible hints normalize to HIP device memory");
    assert_eq!(host_visible.tier(), themis::MemoryTier::Device);

    let registers = device.alloc_zeroed_with_hint::<u32>(
        4,
        themis::PlacementHint::Tier(themis::MemoryTier::Registers),
    );
    assert!(matches!(
        registers,
        Err(HephaestusError::AllocationFailed { message })
            if message == "ROCm primary buffers cannot be allocated from budget-only tier Registers"
    ));
}

#[test]
fn subrange_write_preserves_unwritten_values() {
    let Some(device) = device("subrange_write_preserves_unwritten_values") else {
        return;
    };

    let buffer = device.upload(&[1_u32, 2, 3, 4, 5]).expect("HIP upload");
    device
        .write_sub_buffer(&buffer, 2, &[9_u32, 8])
        .expect("HIP subrange write");
    let mut output = [0_u32; 5];
    device.download(&buffer, &mut output).expect("HIP download");
    assert_eq!(output, [1, 2, 9, 8, 5]);
}

#[test]
fn length_mismatches_are_rejected_before_transfer() {
    let Some(device) = device("length_mismatches_are_rejected_before_transfer") else {
        return;
    };

    let buffer = device.upload(&[1.0_f32, 2.0]).expect("HIP upload");
    let mut output = [0.0_f32; 3];
    assert_length_mismatch(device.download(&buffer, &mut output), 3, 2);
    assert_length_mismatch(device.write_buffer(&buffer, &[1.0_f32]), 1, 2);
}

#[test]
fn empty_buffers_roundtrip_without_hip_allocation() {
    let Some(device) = device("empty_buffers_roundtrip_without_hip_allocation") else {
        return;
    };

    let buffer = device.upload::<u16>(&[]).expect("empty HIP upload");
    assert!(buffer.is_empty());
    let mut output = [];
    device
        .download(&buffer, &mut output)
        .expect("empty HIP download");
}

#[test]
fn elementwise_kernels_match_cpu_values_and_reject_invalid_output_contracts() {
    let Some(device) =
        device("elementwise_kernels_match_cpu_values_and_reject_invalid_output_contracts")
    else {
        return;
    };

    let lhs: Vec<f32> = (0..513).map(|index| index as f32).collect();
    let rhs: Vec<f32> = (0..513).map(|index| (index % 7) as f32).collect();
    let lhs_buffer = device.upload(&lhs).expect("HIP lhs upload");
    let rhs_buffer = device.upload(&rhs).expect("HIP rhs upload");

    let sum = binary_elementwise::<AddOp, _>(&device, &lhs_buffer, &rhs_buffer)
        .expect("HIP binary elementwise add");
    let mut sum_output = vec![0.0_f32; lhs.len()];
    device
        .download(&sum, &mut sum_output)
        .expect("HIP sum download");
    let expected_sum: Vec<f32> = lhs.iter().zip(&rhs).map(|(lhs, rhs)| lhs + rhs).collect();
    assert_eq!(sum_output, expected_sum);

    let negated = unary_elementwise::<NegOp, _>(&device, &lhs_buffer).expect("HIP unary negate");
    let mut negated_output = vec![0.0_f32; lhs.len()];
    device
        .download(&negated, &mut negated_output)
        .expect("HIP negation download");
    let expected_negated: Vec<f32> = lhs.iter().map(|value| -value).collect();
    assert_eq!(negated_output, expected_negated);

    let doubled = scalar_elementwise::<MulOp, _>(&device, &lhs_buffer, 2.0)
        .expect("HIP scalar elementwise multiply");
    let mut doubled_output = vec![0.0_f32; lhs.len()];
    device
        .download(&doubled, &mut doubled_output)
        .expect("HIP scalar multiply download");
    let expected_doubled: Vec<f32> = lhs.iter().map(|value| value * 2.0).collect();
    assert_eq!(doubled_output, expected_doubled);

    let mismatch = device
        .alloc_zeroed::<f32>(lhs.len() - 1)
        .expect("HIP mismatch buffer");
    assert!(matches!(
        binary_elementwise_into::<AddOp, _>(
            &device,
            &lhs_buffer,
            &rhs_buffer,
            &mismatch,
            hephaestus_core::BlockWidth::DEFAULT,
        ),
        Err(HephaestusError::LengthMismatch {
            host_len,
            device_len,
        }) if host_len == lhs.len() - 1 && device_len == lhs.len()
    ));

    assert!(matches!(
        binary_elementwise_into::<AddOp, _>(
            &device,
            &lhs_buffer,
            &rhs_buffer,
            &lhs_buffer,
            hephaestus_core::BlockWidth::DEFAULT,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "output buffer must not alias binary left input"
    ));
}

#[test]
fn reduction_kernels_match_cpu_values_across_tree_passes_and_boundaries() {
    let Some(device) =
        device("reduction_kernels_match_cpu_values_across_tree_passes_and_boundaries")
    else {
        return;
    };

    let input: Vec<u32> = (0..513).map(|index| (index % 17) as u32).collect();
    let input_buffer = device.upload(&input).expect("HIP reduction input upload");
    let width = BlockWidth::new(128).expect("test reduction width is non-zero");

    let sum =
        reduction_with_width::<SumOp, _>(&device, &input_buffer, width).expect("HIP sum reduction");
    let min =
        reduction_with_width::<MinOp, _>(&device, &input_buffer, width).expect("HIP min reduction");
    let max =
        reduction_with_width::<MaxOp, _>(&device, &input_buffer, width).expect("HIP max reduction");

    let expected_sum: u32 = input.iter().copied().sum();
    let expected_min = input.iter().copied().min().expect("non-empty input");
    let expected_max = input.iter().copied().max().expect("non-empty input");
    let mut sum_output = [0_u32];
    let mut min_output = [0_u32];
    let mut max_output = [0_u32];
    device
        .download(&sum, &mut sum_output)
        .expect("HIP sum download");
    device
        .download(&min, &mut min_output)
        .expect("HIP min download");
    device
        .download(&max, &mut max_output)
        .expect("HIP max download");
    assert_eq!(sum_output, [expected_sum]);
    assert_eq!(min_output, [expected_min]);
    assert_eq!(max_output, [expected_max]);

    let empty = device
        .upload::<u32>(&[])
        .expect("HIP empty reduction upload");
    let empty_sum =
        reduction_with_width::<SumOp, _>(&device, &empty, width).expect("HIP empty sum reduction");
    let mut empty_output = [u32::MAX];
    device
        .download(&empty_sum, &mut empty_output)
        .expect("HIP empty sum download");
    assert_eq!(empty_output, [0]);

    let invalid_width = BlockWidth::new(192).expect("test invalid width is non-zero");
    assert!(matches!(
        reduction_with_width::<SumOp, _>(&device, &input_buffer, invalid_width),
        Err(HephaestusError::DispatchFailed { message })
            if message == "reduction block width 192 must be a power of two"
    ));
}

#[test]
fn axis_reduction_kernels_match_cpu_values_and_reject_invalid_layouts() {
    let Some(device) = device("axis_reduction_kernels_match_cpu_values_and_reject_invalid_layouts")
    else {
        return;
    };

    let width = BlockWidth::new(2).expect("test reduction width is non-zero");
    let input: Vec<u32> = vec![1, 2, 3, 4, 10, 20, 30, 40, 5, 7, 9, 11];
    let input_buffer = device.upload(&input).expect("HIP axis input upload");
    let input_layout = Layout::c_contiguous([3, 4]).expect("axis input layout");
    let input_operand = StridedOperand {
        buffer: &input_buffer,
        layout: &input_layout,
    };

    let sum_rows = sum_axis(&device, input_operand, 1, width).expect("HIP axis row sum");
    let sum_columns = sum_axis(&device, input_operand, 0, width).expect("HIP axis column sum");
    let min_rows = min_axis(&device, input_operand, 1, width).expect("HIP axis row min");
    let max_rows = max_axis(&device, input_operand, 1, width).expect("HIP axis row max");
    let mut sum_rows_output = [0_u32; 3];
    let mut sum_columns_output = [0_u32; 4];
    let mut min_rows_output = [0_u32; 3];
    let mut max_rows_output = [0_u32; 3];
    device
        .download(&sum_rows, &mut sum_rows_output)
        .expect("HIP row sum download");
    device
        .download(&sum_columns, &mut sum_columns_output)
        .expect("HIP column sum download");
    device
        .download(&min_rows, &mut min_rows_output)
        .expect("HIP row min download");
    device
        .download(&max_rows, &mut max_rows_output)
        .expect("HIP row max download");
    assert_eq!(sum_rows_output, [10, 100, 32]);
    assert_eq!(sum_columns_output, [16, 29, 42, 55]);
    assert_eq!(min_rows_output, [1, 10, 5]);
    assert_eq!(max_rows_output, [4, 40, 11]);

    let mean_input: Vec<f32> = (1..=12).map(|value| value as f32).collect();
    let mean_buffer = device.upload(&mean_input).expect("HIP mean input upload");
    let mean_layout = Layout::c_contiguous([3, 4]).expect("mean input layout");
    let mean = mean_axis(
        &device,
        StridedOperand {
            buffer: &mean_buffer,
            layout: &mean_layout,
        },
        1,
        width,
    )
    .expect("HIP row mean");
    let mut mean_output = [0.0_f32; 3];
    device
        .download(&mean, &mut mean_output)
        .expect("HIP mean download");
    assert_eq!(mean_output, [2.5, 6.5, 10.5]);

    let wrong_layout = Layout::c_contiguous([3, 2]).expect("wrong output layout");
    let wrong_buffer = device.alloc_zeroed::<u32>(6).expect("wrong output buffer");
    assert!(matches!(
        hephaestus_rocm::sum_axis_into(
            &device,
            input_operand,
            1,
            StridedOperand {
                buffer: &wrong_buffer,
                layout: &wrong_layout,
            },
            width,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("axis reduction output shape mismatch")
    ));

    let output_layout = Layout::c_contiguous([3, 1]).expect("alias output layout");
    assert!(matches!(
        hephaestus_rocm::sum_axis_into(
            &device,
            input_operand,
            1,
            StridedOperand {
                buffer: &input_buffer,
                layout: &output_layout,
            },
            width,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "axis reduction output buffer must not alias input buffer"
    ));

    let empty_buffer = device.upload::<u32>(&[]).expect("empty axis input upload");
    let empty_input_layout = Layout::c_contiguous([3, 0]).expect("empty input layout");
    let empty_output_layout = Layout::c_contiguous([3, 1]).expect("empty output layout");
    let empty_output = device.alloc_zeroed::<u32>(3).expect("empty output buffer");
    assert!(matches!(
        mean_axis_into(
            &device,
            StridedOperand {
                buffer: &empty_buffer,
                layout: &empty_input_layout,
            },
            1,
            StridedOperand {
                buffer: &empty_output,
                layout: &empty_output_layout,
            },
            width,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "mean_axis is undefined for empty axis 1"
    ));
}

#[test]
fn scan_kernels_match_cpu_values_across_axes_directions_and_chunk_boundaries() {
    let Some(device) =
        device("scan_kernels_match_cpu_values_across_axes_directions_and_chunk_boundaries")
    else {
        return;
    };

    let width = BlockWidth::new(2).expect("test scan width is non-zero");
    let input: Vec<i32> = (1..=8).collect();
    let input_buffer = device.upload(&input).expect("HIP scan input upload");
    let input_layout = Layout::c_contiguous([2, 4]).expect("scan input layout");
    let input_operand = StridedOperand {
        buffer: &input_buffer,
        layout: &input_layout,
    };

    let forward = cumsum(&device, input_operand, 1, width).expect("HIP forward cumulative sum");
    let mut forward_output = [0_i32; 8];
    device
        .download(&forward, &mut forward_output)
        .expect("HIP forward scan download");
    assert_eq!(forward_output, [1, 3, 6, 10, 5, 11, 18, 26]);

    let reverse =
        scan_axis::<CumSumOp, _>(&device, input_operand, 1, ScanDirection::Reverse, width)
            .expect("HIP reverse cumulative sum");
    let mut reverse_output = [0_i32; 8];
    device
        .download(&reverse, &mut reverse_output)
        .expect("HIP reverse scan download");
    assert_eq!(reverse_output, [10, 9, 7, 4, 26, 21, 15, 8]);

    let column_scan = cumsum(&device, input_operand, 0, width).expect("HIP column scan");
    let mut column_output = [0_i32; 8];
    device
        .download(&column_scan, &mut column_output)
        .expect("HIP column scan download");
    assert_eq!(column_output, [1, 2, 3, 4, 6, 8, 10, 12]);

    let reverse_product =
        cumprod(&device, input_operand, 1, width).expect("HIP reverse cumulative product");
    let mut product_output = [0_i32; 8];
    device
        .download(&reverse_product, &mut product_output)
        .expect("HIP product scan download");
    assert_eq!(product_output, [24, 24, 12, 4, 1680, 336, 56, 8]);

    let long_input: Vec<i32> = (0..1_025).map(|index| index % 7 - 3).collect();
    let long_buffer = device.upload(&long_input).expect("HIP long scan upload");
    let long_layout = Layout::c_contiguous([1, 1_025]).expect("long scan layout");
    let long_output = scan_axis::<CumSumOp, _>(
        &device,
        StridedOperand {
            buffer: &long_buffer,
            layout: &long_layout,
        },
        1,
        ScanDirection::Forward,
        width,
    )
    .expect("HIP long scan");
    let mut got_long = vec![0_i32; long_input.len()];
    device
        .download(&long_output, &mut got_long)
        .expect("HIP long scan download");
    let expected_long: Vec<i32> = long_input
        .iter()
        .scan(0_i32, |acc, value| {
            *acc += *value;
            Some(*acc)
        })
        .collect();
    assert_eq!(got_long, expected_long);

    let wrong_layout = Layout::c_contiguous([4, 2]).expect("wrong scan output layout");
    let wrong_buffer = device
        .alloc_zeroed::<i32>(8)
        .expect("wrong scan output buffer");
    assert!(matches!(
        scan_axis_into::<CumSumOp, _>(
            &device,
            input_operand,
            1,
            ScanDirection::Forward,
            StridedOperand {
                buffer: &wrong_buffer,
                layout: &wrong_layout,
            },
            width,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("scan output shape mismatch")
    ));

    assert!(matches!(
        scan_axis_into::<CumSumOp, _>(
            &device,
            input_operand,
            1,
            ScanDirection::Forward,
            StridedOperand {
                buffer: &input_buffer,
                layout: &input_layout,
            },
            width,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "scan output buffer must not alias input buffer"
    ));
}

#[test]
fn matmul_kernel_matches_cpu_values_across_tile_boundaries_and_rejects_invalid_contracts() {
    let Some(device) = device(
        "matmul_kernel_matches_cpu_values_across_tile_boundaries_and_rejects_invalid_contracts",
    ) else {
        return;
    };

    let lhs_values: Vec<i32> = (0..17 * 19).map(|index| (index % 7) - 3).collect();
    let rhs_values: Vec<i32> = (0..19 * 5).map(|index| (index % 5) - 2).collect();
    let lhs_buffer = device.upload(&lhs_values).expect("HIP matmul lhs upload");
    let rhs_buffer = device.upload(&rhs_values).expect("HIP matmul rhs upload");
    let lhs_layout = Layout::c_contiguous([17, 19]).expect("matmul lhs layout");
    let rhs_layout = Layout::c_contiguous([19, 5]).expect("matmul rhs layout");
    let lhs = StridedOperand {
        buffer: &lhs_buffer,
        layout: &lhs_layout,
    };
    let rhs = StridedOperand {
        buffer: &rhs_buffer,
        layout: &rhs_layout,
    };

    let output = matmul(&device, lhs, rhs).expect("HIP tiled matmul");
    let mut output_values = vec![0_i32; 17 * 5];
    device
        .download(&output, &mut output_values)
        .expect("HIP matmul download");
    let mut expected = Vec::with_capacity(17 * 5);
    for row in 0..17 {
        for col in 0..5 {
            expected.push(
                (0..19)
                    .map(|shared| lhs_values[row * 19 + shared] * rhs_values[shared * 5 + col])
                    .sum(),
            );
        }
    }
    assert_eq!(output_values, expected);

    let output_layout = Layout::c_contiguous([17, 5]).expect("matmul output layout");
    let output_into = device
        .alloc_zeroed::<i32>(17 * 5)
        .expect("HIP caller-owned matmul output");
    matmul_into(
        &device,
        lhs,
        rhs,
        StridedOperand {
            buffer: &output_into,
            layout: &output_layout,
        },
    )
    .expect("HIP caller-owned tiled matmul");
    let mut output_into_values = vec![0_i32; 17 * 5];
    device
        .download(&output_into, &mut output_into_values)
        .expect("HIP caller-owned matmul download");
    assert_eq!(output_into_values, expected);

    let wrong_layout = Layout::c_contiguous([17, 4]).expect("wrong matmul output layout");
    let wrong_output = device
        .alloc_zeroed::<i32>(17 * 4)
        .expect("wrong matmul output buffer");
    assert!(matches!(
        matmul_into(
            &device,
            lhs,
            rhs,
            StridedOperand {
                buffer: &wrong_output,
                layout: &wrong_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("matmul dimension mismatch")
    ));

    assert!(matches!(
        matmul_into(
            &device,
            lhs,
            rhs,
            StridedOperand {
                buffer: &lhs_buffer,
                layout: &lhs_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "output buffer must not alias either input buffer"
    ));
}

#[test]
fn batched_matmul_kernel_matches_cpu_values_with_broadcast_and_rejects_invalid_contracts() {
    let Some(device) = device(
        "batched_matmul_kernel_matches_cpu_values_with_broadcast_and_rejects_invalid_contracts",
    ) else {
        return;
    };

    let lhs_values: Vec<i32> = (0..3 * 17 * 19).map(|index| (index % 7) - 3).collect();
    let rhs_values: Vec<i32> = (0..19 * 5).map(|index| (index % 5) - 2).collect();
    let lhs_buffer = device.upload(&lhs_values).expect("HIP batched lhs upload");
    let rhs_buffer = device
        .upload(&rhs_values)
        .expect("HIP broadcast rhs upload");
    let lhs_layout = Layout::c_contiguous([3, 17, 19]).expect("batched lhs layout");
    let rhs_layout = Layout::c_contiguous([1, 19, 5]).expect("broadcast rhs layout");
    let lhs = StridedOperand {
        buffer: &lhs_buffer,
        layout: &lhs_layout,
    };
    let rhs = StridedOperand {
        buffer: &rhs_buffer,
        layout: &rhs_layout,
    };

    let output = batched_matmul(&device, lhs, rhs).expect("HIP batched matmul");
    let mut output_values = vec![0_i32; 3 * 17 * 5];
    device
        .download(&output, &mut output_values)
        .expect("HIP batched matmul download");
    let mut expected = Vec::with_capacity(3 * 17 * 5);
    for batch in 0..3 {
        for row in 0..17 {
            for col in 0..5 {
                expected.push(
                    (0..19)
                        .map(|shared| {
                            lhs_values[(batch * 17 + row) * 19 + shared]
                                * rhs_values[shared * 5 + col]
                        })
                        .sum(),
                );
            }
        }
    }
    assert_eq!(output_values, expected);

    let output_layout = Layout::c_contiguous([3, 17, 5]).expect("batched output layout");
    let output_into = device
        .alloc_zeroed::<i32>(3 * 17 * 5)
        .expect("HIP caller-owned batched output");
    batched_matmul_into(
        &device,
        lhs,
        rhs,
        StridedOperand {
            buffer: &output_into,
            layout: &output_layout,
        },
    )
    .expect("HIP caller-owned batched matmul");
    let mut output_into_values = vec![0_i32; 3 * 17 * 5];
    device
        .download(&output_into, &mut output_into_values)
        .expect("HIP caller-owned batched download");
    assert_eq!(output_into_values, expected);

    let wrong_layout = Layout::c_contiguous([2, 17, 5]).expect("wrong batched output layout");
    let wrong_output = device
        .alloc_zeroed::<i32>(2 * 17 * 5)
        .expect("wrong batched output buffer");
    assert!(matches!(
        batched_matmul_into(
            &device,
            lhs,
            rhs,
            StridedOperand {
                buffer: &wrong_output,
                layout: &wrong_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("batched matmul shape mismatch")
    ));

    assert!(matches!(
        batched_matmul_into(
            &device,
            lhs,
            rhs,
            StridedOperand {
                buffer: &lhs_buffer,
                layout: &lhs_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "output buffer must not alias either input buffer"
    ));
}
