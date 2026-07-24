//! Value-semantic contracts for the ROCm device substrate.
//!
//! The default build verifies the typed unavailable path. With `--features
//! rocm`, the same tests execute against HIP when an AMD device is present.
//! Hardware CI sets `HEPHAESTUS_ROCM_REQUIRE_DEVICE=1` so an unavailable device
//! fails that lane instead of being reported as device evidence.

use hephaestus_core::{
    AddOp, BinaryStorageKernel, Binding, BindingDecl, BlockWidth, CommandStream, ComputeDevice,
    ComputeDeviceCapabilities, DeviceBuffer, DeviceFeature, DispatchGrid, GroupedBinding,
    GroupedBindingDecl, GroupedCommandStream, GroupedKernelDevice, GroupedKernelInterface,
    GroupedKernelSource, HephaestusError, HipC, IdentityOp, KernelDevice, KernelInterface,
    KernelSource, MaxOp, MinOp, MulOp, NegOp, SumOp,
};
use hephaestus_rocm::{
    CumSumOp, GpuCsrMatrix, Result, RocmDevice, RocmMultiStorageKernel, ScanDirection,
    StridedOperand, batched_matmul, batched_matmul_into, binary_elementwise,
    binary_elementwise_into, binary_elementwise_strided, binary_elementwise_strided_into, cumprod,
    cumsum, det, dot, kron, kron_into, matmul, matmul_into, matpow, matrix_rank,
    matrix_rank_with_tolerance, max_axis, mean_axis, mean_axis_into, min_axis, norm_l1, norm_l2,
    norm_max, normal_with_seed, reduction_with_width, scalar_elementwise,
    scalar_elementwise_strided_into, scan_axis, scan_axis_into, spmm, spmm_into, spmv, spmv_many,
    spmv_many_into, sum_axis, trace, unary_elementwise, unary_elementwise_strided,
    unary_elementwise_strided_into, uniform_with_seed,
};
use leto::Layout;
use std::borrow::Cow;

struct StreamAddKernel;

impl KernelInterface for StreamAddKernel {
    type Params = u32;

    const LABEL: &'static str = "rocm-contract-stream-add";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [64, 1, 1];
}

impl KernelSource<HipC> for StreamAddKernel {
    const ENTRY: &'static str = "rocm_contract_stream_add";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r#"
extern "C" __global__ void rocm_contract_stream_add(
    const float* lhs,
    const float* rhs,
    float* out,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = lhs[i] + rhs[i];
}
"#,
        )
    }
}

struct GroupedStreamAddKernel;

impl GroupedKernelInterface for GroupedStreamAddKernel {
    type Params = u32;

    const LABEL: &'static str = "rocm-contract-grouped-stream-add";
    const BINDINGS: &'static [GroupedBindingDecl] = &[
        GroupedBindingDecl::read_only::<f32>(0, 0),
        GroupedBindingDecl::read_only::<f32>(0, 1),
        GroupedBindingDecl::read_write::<f32>(0, 2),
    ];
    const PARAM_GROUP: u32 = 0;
    const PARAM_BINDING: u32 = 3;
    const WORKGROUP: [u32; 3] = [64, 1, 1];
}

impl GroupedKernelSource<HipC> for GroupedStreamAddKernel {
    const ENTRY: &'static str = "rocm_contract_grouped_stream_add";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r#"
extern "C" __global__ void rocm_contract_grouped_stream_add(
    const float* lhs,
    const float* rhs,
    float* out,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = lhs[i] + rhs[i];
}
"#,
        )
    }
}

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
fn strided_elementwise_kernels_match_cpu_values_and_reject_invalid_layouts() {
    let Some(device) =
        device("strided_elementwise_kernels_match_cpu_values_and_reject_invalid_layouts")
    else {
        return;
    };

    let width = BlockWidth::DEFAULT;
    let lhs_values = [99_i32, 1, 2, 3, 4];
    let rhs_values = [10_i32, 20];
    let lhs_buffer = device.upload(&lhs_values).expect("HIP strided lhs upload");
    let rhs_buffer = device
        .upload(&rhs_values)
        .expect("HIP broadcast rhs upload");
    let lhs_layout = Layout::new([2, 2], [1, 2], 1);
    let rhs_layout = Layout::c_contiguous([1, 2]).expect("broadcast rhs layout");
    let lhs = StridedOperand {
        buffer: &lhs_buffer,
        layout: &lhs_layout,
    };
    let rhs = StridedOperand {
        buffer: &rhs_buffer,
        layout: &rhs_layout,
    };

    let sum = binary_elementwise_strided::<AddOp, _, 2>(&device, lhs, rhs, [2, 2], width)
        .expect("HIP strided add");
    let mut sum_values = [0_i32; 4];
    device
        .download(&sum, &mut sum_values)
        .expect("HIP strided add download");
    assert_eq!(sum_values, [11, 23, 12, 24]);

    let identity = unary_elementwise_strided::<IdentityOp, _, 2>(&device, lhs, [2, 2], width)
        .expect("HIP strided identity");
    let mut identity_values = [0_i32; 4];
    device
        .download(&identity, &mut identity_values)
        .expect("HIP strided identity download");
    assert_eq!(identity_values, [1, 3, 2, 4]);

    let scalar_output_layout = Layout::c_contiguous([2, 2]).expect("strided scalar layout");
    let scalar_output = device
        .alloc_zeroed::<i32>(4)
        .expect("strided scalar output");
    scalar_elementwise_strided_into::<MulOp, _, 2>(
        &device,
        lhs,
        2,
        StridedOperand {
            buffer: &scalar_output,
            layout: &scalar_output_layout,
        },
        width,
    )
    .expect("HIP strided scalar");
    let mut scalar_values = [0_i32; 4];
    device
        .download(&scalar_output, &mut scalar_values)
        .expect("HIP strided scalar download");
    assert_eq!(scalar_values, [2, 6, 4, 8]);

    let output_into = device
        .alloc_zeroed::<i32>(4)
        .expect("strided caller-owned output");
    binary_elementwise_strided_into::<AddOp, _, 2>(
        &device,
        lhs,
        rhs,
        StridedOperand {
            buffer: &output_into,
            layout: &scalar_output_layout,
        },
        width,
    )
    .expect("HIP caller-owned strided add");
    let mut output_into_values = [0_i32; 4];
    device
        .download(&output_into, &mut output_into_values)
        .expect("HIP caller-owned strided add download");
    assert_eq!(output_into_values, [11, 23, 12, 24]);

    let empty_buffer = device.upload::<i32>(&[]).expect("empty strided upload");
    let empty_layout = Layout::c_contiguous([0, 2]).expect("empty strided layout");
    let empty = unary_elementwise_strided::<IdentityOp, _, 2>(
        &device,
        StridedOperand {
            buffer: &empty_buffer,
            layout: &empty_layout,
        },
        [0, 2],
        width,
    )
    .expect("empty strided identity");
    assert_eq!(empty.len(), 0);

    let bad_rhs_values = [1_i32; 6];
    let bad_rhs_buffer = device
        .upload(&bad_rhs_values)
        .expect("bad broadcast upload");
    let bad_rhs_layout = Layout::c_contiguous([3, 2]).expect("bad broadcast layout");
    assert!(matches!(
        binary_elementwise_strided_into::<AddOp, _, 2>(
            &device,
            lhs,
            StridedOperand {
                buffer: &bad_rhs_buffer,
                layout: &bad_rhs_layout,
            },
            StridedOperand {
                buffer: &output_into,
                layout: &scalar_output_layout,
            },
            width,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("layout rejected:")
    ));

    assert!(matches!(
        unary_elementwise_strided_into::<IdentityOp, _, 2>(
            &device,
            lhs,
            StridedOperand {
                buffer: &lhs_buffer,
                layout: &lhs_layout,
            },
            width,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "output buffer must not alias input buffer"
    ));

    let aliased_layout = Layout::new([2, 2], [0, 1], 0);
    let aliased_output = device.alloc_zeroed::<i32>(2).expect("zero-stride output");
    assert!(matches!(
        unary_elementwise_strided_into::<IdentityOp, _, 2>(
            &device,
            lhs,
            StridedOperand {
                buffer: &aliased_output,
                layout: &aliased_layout,
            },
            width,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "output layout must not contain zero-stride aliasing"
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

#[test]
fn kron_kernel_matches_cpu_values_for_strided_operands_and_rejects_invalid_contracts() {
    let Some(device) =
        device("kron_kernel_matches_cpu_values_for_strided_operands_and_rejects_invalid_contracts")
    else {
        return;
    };

    let lhs_values = [99_i32, 1, 2, 3, 4];
    let rhs_values = [5_i32, 6, 7, 8];
    let lhs_buffer = device.upload(&lhs_values).expect("HIP kron lhs upload");
    let rhs_buffer = device.upload(&rhs_values).expect("HIP kron rhs upload");
    let lhs_layout = Layout::new([2, 2], [2, 1], 1);
    let rhs_layout = Layout::c_contiguous([2, 2]).expect("kron rhs layout");
    let lhs = StridedOperand {
        buffer: &lhs_buffer,
        layout: &lhs_layout,
    };
    let rhs = StridedOperand {
        buffer: &rhs_buffer,
        layout: &rhs_layout,
    };
    let expected = vec![
        5_i32, 6, 10, 12, 7, 8, 14, 16, 15, 18, 20, 24, 21, 24, 28, 32,
    ];

    let output = kron(&device, lhs, rhs).expect("HIP kron");
    let mut output_values = vec![0_i32; 16];
    device
        .download(&output, &mut output_values)
        .expect("HIP kron download");
    assert_eq!(output_values, expected);

    let strided_output_layout = Layout::new([4, 4], [8, 2], 0);
    let strided_output = device
        .alloc_zeroed::<i32>(31)
        .expect("HIP strided kron output");
    kron_into(
        &device,
        lhs,
        rhs,
        StridedOperand {
            buffer: &strided_output,
            layout: &strided_output_layout,
        },
    )
    .expect("HIP caller-owned strided kron");
    let mut strided_values = vec![0_i32; 31];
    device
        .download(&strided_output, &mut strided_values)
        .expect("HIP strided kron download");
    let gathered: Vec<_> = strided_values
        .chunks_exact(8)
        .take(4)
        .flat_map(|row| row.iter().step_by(2).take(4).copied())
        .collect();
    assert_eq!(gathered, expected);

    let empty_lhs_buffer = device.upload(&[]).expect("HIP empty kron lhs upload");
    let empty_lhs_layout = Layout::c_contiguous([0, 2]).expect("empty kron lhs layout");
    let empty_output = kron(
        &device,
        StridedOperand {
            buffer: &empty_lhs_buffer,
            layout: &empty_lhs_layout,
        },
        rhs,
    )
    .expect("HIP empty kron");
    assert_eq!(empty_output.len(), 0);

    let wrong_layout = Layout::c_contiguous([3, 4]).expect("wrong kron output layout");
    let wrong_output = device
        .alloc_zeroed::<i32>(12)
        .expect("wrong kron output buffer");
    assert!(matches!(
        kron_into(
            &device,
            lhs,
            rhs,
            StridedOperand {
                buffer: &wrong_output,
                layout: &wrong_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("Kronecker output shape mismatch")
    ));

    assert!(matches!(
        kron_into(
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

    let aliased_output_layout = Layout::new([4, 4], [0, 1], 0);
    let aliased_output = device
        .alloc_zeroed::<i32>(4)
        .expect("zero-stride kron output buffer");
    assert!(matches!(
        kron_into(
            &device,
            lhs,
            rhs,
            StridedOperand {
                buffer: &aliased_output,
                layout: &aliased_output_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "Kronecker output layout must not contain zero-stride aliasing"
    ));
}

#[test]
fn matpow_matches_cpu_values_for_strided_inputs_and_rejects_non_square() {
    let Some(device) =
        device("matpow_matches_cpu_values_for_strided_inputs_and_rejects_non_square")
    else {
        return;
    };

    let shear_values = [1_i32, 1, 0, 1];
    let shear_buffer = device
        .upload(&shear_values)
        .expect("HIP matpow shear upload");
    let square_layout = Layout::c_contiguous([2, 2]).expect("matpow square layout");
    let shear = matpow(
        &device,
        StridedOperand {
            buffer: &shear_buffer,
            layout: &square_layout,
        },
        5,
    )
    .expect("HIP matpow shear");
    let mut shear_output = [0_i32; 4];
    device
        .download(&shear, &mut shear_output)
        .expect("HIP matpow shear download");
    assert_eq!(shear_output, [1, 5, 0, 1]);

    let strided_values = [99_i32, 1, 2, 3, 4];
    let strided_buffer = device
        .upload(&strided_values)
        .expect("HIP strided matpow upload");
    let strided_layout = Layout::new([2, 2], [1, 2], 1);
    let strided_power = matpow(
        &device,
        StridedOperand {
            buffer: &strided_buffer,
            layout: &strided_layout,
        },
        2,
    )
    .expect("HIP strided matpow");
    let mut strided_output = [0_i32; 4];
    device
        .download(&strided_power, &mut strided_output)
        .expect("HIP strided matpow download");
    assert_eq!(strided_output, [7, 15, 10, 22]);

    let identity_power = matpow(
        &device,
        StridedOperand {
            buffer: &strided_buffer,
            layout: &strided_layout,
        },
        0,
    )
    .expect("HIP matpow identity");
    let mut identity_output = [0_i32; 4];
    device
        .download(&identity_power, &mut identity_output)
        .expect("HIP matpow identity download");
    assert_eq!(identity_output, [1, 0, 0, 1]);

    let nonsquare_values = [1_i32, 2, 3, 4, 5, 6];
    let nonsquare_buffer = device
        .upload(&nonsquare_values)
        .expect("HIP nonsquare matpow upload");
    let nonsquare_layout = Layout::c_contiguous([2, 3]).expect("nonsquare matpow layout");
    assert!(matches!(
        matpow(
            &device,
            StridedOperand {
                buffer: &nonsquare_buffer,
                layout: &nonsquare_layout,
            },
            2,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "matpow requires a square matrix, got shape [2, 3]"
    ));
}

#[test]
fn matrix_rank_and_det_match_cpu_values_and_tolerance_contracts() {
    let Some(device) = device("matrix_rank_and_det_match_cpu_values_and_tolerance_contracts")
    else {
        return;
    };

    let values = [99.0_f32, 1.0, 2.0, 3.0, 4.0];
    let buffer = device.upload(&values).expect("HIP matrix-rank upload");
    let layout = Layout::new([2, 2], [1, 2], 1);
    let matrix = StridedOperand {
        buffer: &buffer,
        layout: &layout,
    };
    assert_eq!(matrix_rank(&device, matrix).expect("HIP matrix rank"), 2);

    let determinant = det(&device, matrix).expect("HIP determinant");
    let mut determinant_value = [0.0_f32];
    device
        .download(&determinant, &mut determinant_value)
        .expect("HIP determinant download");
    assert_eq!(determinant_value, [-2.0]);

    let tolerance_values = [1.0_f32, 0.0, 0.0, 1.0e-10];
    let tolerance_buffer = device
        .upload(&tolerance_values)
        .expect("HIP matrix-rank tolerance upload");
    let tolerance_layout = Layout::c_contiguous([2, 2]).expect("matrix-rank tolerance layout");
    let tolerance_matrix = StridedOperand {
        buffer: &tolerance_buffer,
        layout: &tolerance_layout,
    };
    assert_eq!(
        matrix_rank(&device, tolerance_matrix).expect("default matrix rank"),
        1
    );
    assert_eq!(
        matrix_rank_with_tolerance(&device, tolerance_matrix, 1.0e-12).expect("strict matrix rank"),
        2
    );

    let singular_values = [1.0_f32, 2.0, 2.0, 4.0];
    let singular_buffer = device
        .upload(&singular_values)
        .expect("HIP singular upload");
    let singular = StridedOperand {
        buffer: &singular_buffer,
        layout: &tolerance_layout,
    };
    assert_eq!(
        matrix_rank(&device, singular).expect("singular matrix rank"),
        1
    );
    let singular_determinant = det(&device, singular).expect("singular determinant");
    let mut singular_value = [1.0_f32];
    device
        .download(&singular_determinant, &mut singular_value)
        .expect("singular determinant download");
    assert_eq!(singular_value, [0.0]);

    let nonsquare_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let nonsquare_buffer = device.upload(&nonsquare_values).expect("nonsquare upload");
    let nonsquare_layout = Layout::c_contiguous([2, 3]).expect("nonsquare layout");
    assert!(matches!(
        det(
            &device,
            StridedOperand {
                buffer: &nonsquare_buffer,
                layout: &nonsquare_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "det requires a square matrix, got shape [2, 3]"
    ));
}

#[test]
fn seeded_random_initializers_match_determinism_and_distribution_contracts() {
    let Some(device) =
        device("seeded_random_initializers_match_determinism_and_distribution_contracts")
    else {
        return;
    };

    let shape = [1000];
    let uniform =
        uniform_with_seed(&device, shape, -2.0_f32, 5.0, 42).expect("HIP uniform initializer");
    let mut uniform_values = vec![0.0_f32; 1000];
    device
        .download(&uniform, &mut uniform_values)
        .expect("HIP uniform download");
    let uniform_again = uniform_with_seed(&device, shape, -2.0_f32, 5.0, 42)
        .expect("HIP repeated uniform initializer");
    let mut uniform_again_values = vec![0.0_f32; 1000];
    device
        .download(&uniform_again, &mut uniform_again_values)
        .expect("HIP repeated uniform download");
    assert_eq!(uniform_values, uniform_again_values);
    assert!(
        uniform_values
            .iter()
            .all(|&value| (-2.0..5.0).contains(&value))
    );

    let normal =
        normal_with_seed(&device, shape, 0.0_f32, 1.0, 42).expect("HIP normal initializer");
    let mut normal_values = vec![0.0_f32; 1000];
    device
        .download(&normal, &mut normal_values)
        .expect("HIP normal download");
    assert!(normal_values.iter().any(|&value| value != 0.0));
}

#[test]
fn sparse_csr_products_match_cpu_values_and_reject_wrong_shapes() {
    let Some(device) = device("sparse_csr_products_match_cpu_values_and_reject_wrong_shapes")
    else {
        return;
    };

    let cpu_csr = leto_ops::CsrMatrix::from_parts(
        vec![2.0_f32, -1.0, 3.0, 4.0],
        vec![0, 2, 1, 2],
        vec![0, 2, 3, 4],
        3,
        3,
    )
    .expect("valid CSR contract fixture");
    let gpu_csr = GpuCsrMatrix::from_cpu(&device, &cpu_csr).expect("HIP CSR upload");
    assert_eq!(gpu_csr.shape(), (3, 3));
    assert_eq!(gpu_csr.nnz(), 4);
    assert_eq!(gpu_csr.to_cpu(&device).expect("HIP CSR download"), cpu_csr);

    let x = device
        .upload(&[1.0_f32, 2.0, 3.0])
        .expect("SpMV input upload");
    let y = spmv(&device, &gpu_csr, &x).expect("HIP SpMV");
    let mut y_values = [0.0_f32; 3];
    device.download(&y, &mut y_values).expect("SpMV download");
    assert_eq!(y_values, [-1.0, 6.0, 12.0]);

    let b = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("SpMM input upload");
    let b_layout = Layout::c_contiguous([3, 2]).expect("SpMM input layout");
    let b_operand = StridedOperand {
        buffer: &b,
        layout: &b_layout,
    };
    let c = spmm(&device, &gpu_csr, b_operand).expect("HIP SpMM");
    let mut c_values = [0.0_f32; 6];
    device.download(&c, &mut c_values).expect("SpMM download");
    assert_eq!(c_values, [-3.0, -2.0, 9.0, 12.0, 20.0, 24.0]);

    let many = spmv_many(&device, &gpu_csr, b_operand).expect("HIP batched SpMV");
    let mut many_values = [0.0_f32; 6];
    device
        .download(&many, &mut many_values)
        .expect("batched SpMV download");
    assert_eq!(many_values, c_values);

    let mut c_reused = device.upload(&[99.0_f32; 6]).expect("SpMM output upload");
    spmm_into(&device, &gpu_csr, b_operand, &mut c_reused).expect("HIP SpMM into");
    let mut reused_values = [0.0_f32; 6];
    device
        .download(&c_reused, &mut reused_values)
        .expect("reused SpMM download");
    assert_eq!(reused_values, c_values);

    let mut many_reused = device
        .upload(&[88.0_f32; 6])
        .expect("batched output upload");
    spmv_many_into(&device, &gpu_csr, b_operand, &mut many_reused).expect("HIP batched SpMV into");
    let mut many_reused_values = [0.0_f32; 6];
    device
        .download(&many_reused, &mut many_reused_values)
        .expect("reused batched SpMV download");
    assert_eq!(many_reused_values, c_values);

    let wrong_x = device.upload(&[1.0_f32, 2.0]).expect("wrong SpMV upload");
    assert_length_mismatch(spmv(&device, &gpu_csr, &wrong_x), 3, 2);
}

#[test]
fn multi_storage_binary_kernel_matches_values_and_rejects_wrong_lengths() {
    let Some(device) =
        device("multi_storage_binary_kernel_matches_values_and_rejects_wrong_lengths")
    else {
        return;
    };

    let kernel = RocmMultiStorageKernel::new(
        "contract-binary",
        r#"
extern "C" __global__ void contract_binary(
    const float* lhs,
    const float* rhs,
    float* out,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = lhs[i] + rhs[i];
}
"#,
        "contract_binary",
        &[0, 1, 2],
        [64, 1, 1],
        0,
    )
    .expect("valid ROCm multi-storage kernel");
    let lhs = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0])
        .expect("multi-storage lhs upload");
    let rhs = device
        .upload(&[5.0_f32, 6.0, 7.0, 8.0])
        .expect("multi-storage rhs upload");
    let output = device
        .alloc_zeroed::<f32>(4)
        .expect("multi-storage output allocation");
    let grid = DispatchGrid::covering_domain([4, 1, 1], [64, 1, 1]).expect("valid dispatch grid");

    <RocmMultiStorageKernel as BinaryStorageKernel<RocmDevice, f32, u32>>::dispatch(
        &kernel, &device, &lhs, &rhs, &output, &4, grid,
    )
    .expect("HIP multi-storage dispatch");
    let mut values = [0.0_f32; 4];
    device
        .download(&output, &mut values)
        .expect("multi-storage output download");
    assert_eq!(values, [6.0, 8.0, 10.0, 12.0]);

    let short_rhs = device
        .upload(&[5.0_f32, 6.0, 7.0])
        .expect("short multi-storage rhs upload");
    assert_length_mismatch(
        <RocmMultiStorageKernel as BinaryStorageKernel<RocmDevice, f32, u32>>::dispatch(
            &kernel, &device, &lhs, &short_rhs, &output, &4, grid,
        ),
        4,
        3,
    );
}

#[test]
fn authored_kernel_streams_preserve_dispatch_copy_fill_and_grouped_values() {
    let Some(device) =
        device("authored_kernel_streams_preserve_dispatch_copy_fill_and_grouped_values")
    else {
        return;
    };

    let lhs = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0])
        .expect("stream lhs upload");
    let rhs = device
        .upload(&[5.0_f32, 6.0, 7.0, 8.0])
        .expect("stream rhs upload");
    let output = device
        .alloc_zeroed::<f32>(4)
        .expect("stream output allocation");
    let grid = DispatchGrid::covering_domain([4, 1, 1], [64, 1, 1]).expect("stream grid");
    let prepared = device.prepare(&StreamAddKernel).expect("stream prepare");
    let bindings = [
        Binding::read(&lhs),
        Binding::read(&rhs),
        Binding::read_write(&output),
    ];
    let mut stream = device.stream().expect("stream open");
    stream
        .encode(&prepared, &bindings, &4, grid)
        .expect("stream dispatch");

    let copied = device
        .alloc_zeroed::<f32>(4)
        .expect("copy output allocation");
    let prefix = device
        .alloc_zeroed::<f32>(4)
        .expect("prefix output allocation");
    stream.copy(&output, &copied).expect("stream copy");
    stream
        .copy_prefix(&output, &prefix, 2)
        .expect("stream prefix copy");
    stream.fill_zero(&output).expect("stream fill");
    stream.submit().expect("stream submit");

    let mut output_values = [9.0_f32; 4];
    let mut copied_values = [0.0_f32; 4];
    let mut prefix_values = [0.0_f32; 4];
    device
        .download(&output, &mut output_values)
        .expect("stream output download");
    device
        .download(&copied, &mut copied_values)
        .expect("stream copy download");
    device
        .download(&prefix, &mut prefix_values)
        .expect("stream prefix download");
    assert_eq!(output_values, [0.0; 4]);
    assert_eq!(copied_values, [6.0, 8.0, 10.0, 12.0]);
    assert_eq!(prefix_values, [6.0, 8.0, 0.0, 0.0]);

    let grouped_prepared = device
        .prepare_grouped(&GroupedStreamAddKernel)
        .expect("grouped stream prepare");
    let grouped_output = device
        .alloc_zeroed::<f32>(4)
        .expect("grouped output allocation");
    let grouped_bindings = [
        GroupedBinding::read(0, 0, &lhs),
        GroupedBinding::read(0, 1, &rhs),
        GroupedBinding::read_write(0, 2, &grouped_output),
    ];
    let mut grouped_stream = device.grouped_stream().expect("grouped stream open");
    grouped_stream
        .encode_grouped_sequence("grouped-contract", |sequence| {
            sequence.encode_grouped(&grouped_prepared, &grouped_bindings, &4, grid)
        })
        .expect("grouped stream dispatch");
    grouped_stream
        .submit_grouped()
        .expect("grouped stream submit");

    let mut grouped_values = [0.0_f32; 4];
    device
        .download(&grouped_output, &mut grouped_values)
        .expect("grouped output download");
    assert_eq!(grouped_values, [6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn map_reductions_match_cpu_values_for_strided_views_and_reject_invalid_shapes() {
    let Some(device) =
        device("map_reductions_match_cpu_values_for_strided_views_and_reject_invalid_shapes")
    else {
        return;
    };

    let matrix_values = [-3_i32, 4, -5, 6, -7, 8];
    let matrix_buffer = device
        .upload(&matrix_values)
        .expect("HIP norm input upload");
    let matrix_layout = Layout::c_contiguous([2, 3]).expect("norm matrix layout");
    let matrix = StridedOperand {
        buffer: &matrix_buffer,
        layout: &matrix_layout,
    };

    let l1 = norm_l1(&device, matrix).expect("HIP L1 norm");
    let mut l1_value = [0_i32];
    device
        .download(&l1, &mut l1_value)
        .expect("HIP L1 download");
    assert_eq!(l1_value, [33]);

    let max = norm_max(&device, matrix).expect("HIP max norm");
    let mut max_value = [0_i32];
    device
        .download(&max, &mut max_value)
        .expect("HIP max norm download");
    assert_eq!(max_value, [8]);

    let dot_left_values = [99_i32, 1, 2, 3, 4, 5];
    let dot_right_values = [6_i32, 7, 8, 9, 10];
    let dot_left_buffer = device
        .upload(&dot_left_values)
        .expect("HIP strided dot lhs upload");
    let dot_right_buffer = device
        .upload(&dot_right_values)
        .expect("HIP strided dot rhs upload");
    let dot_left_layout = Layout::new([5], [1], 1);
    let dot_right_layout = Layout::c_contiguous([5]).expect("dot rhs layout");
    let dot_value_buffer = dot(
        &device,
        StridedOperand {
            buffer: &dot_left_buffer,
            layout: &dot_left_layout,
        },
        StridedOperand {
            buffer: &dot_right_buffer,
            layout: &dot_right_layout,
        },
    )
    .expect("HIP strided dot");
    let mut dot_value = [0_i32];
    device
        .download(&dot_value_buffer, &mut dot_value)
        .expect("HIP dot download");
    assert_eq!(dot_value, [130]);

    let trace_values = [1_i32, 2, 3, 4, 5, 6, 7, 8, 9];
    let trace_buffer = device.upload(&trace_values).expect("HIP trace upload");
    let trace_layout = Layout::c_contiguous([3, 3]).expect("trace layout");
    let trace_value_buffer = trace(
        &device,
        StridedOperand {
            buffer: &trace_buffer,
            layout: &trace_layout,
        },
    )
    .expect("HIP trace");
    let mut trace_value = [0_i32];
    device
        .download(&trace_value_buffer, &mut trace_value)
        .expect("HIP trace download");
    assert_eq!(trace_value, [15]);

    let l2_values = [3.0_f32, 4.0];
    let l2_buffer = device.upload(&l2_values).expect("HIP L2 input upload");
    let l2_layout = Layout::c_contiguous([2]).expect("L2 layout");
    let l2 = norm_l2(
        &device,
        StridedOperand {
            buffer: &l2_buffer,
            layout: &l2_layout,
        },
    )
    .expect("HIP L2 norm");
    let mut l2_value = [0.0_f32];
    device
        .download(&l2, &mut l2_value)
        .expect("HIP L2 download");
    assert_eq!(l2_value, [5.0]);

    let wrong_dot_layout = Layout::c_contiguous([4]).expect("wrong dot layout");
    assert!(matches!(
        dot(
            &device,
            StridedOperand {
                buffer: &dot_left_buffer,
                layout: &dot_left_layout,
            },
            StridedOperand {
                buffer: &dot_right_buffer,
                layout: &wrong_dot_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("dot product shape mismatch")
    ));

    let rectangular_layout = Layout::c_contiguous([2, 3]).expect("rectangular trace layout");
    assert!(matches!(
        trace(
            &device,
            StridedOperand {
                buffer: &matrix_buffer,
                layout: &rectangular_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("trace requires a square matrix")
    ));
}
