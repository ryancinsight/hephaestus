use super::*;

#[test]
fn elementwise_marker_types_are_exported_from_rocm_root() {
    let markers = [
        std::any::TypeId::of::<AbsOp>(),
        std::any::TypeId::of::<AddOp>(),
        std::any::TypeId::of::<CosOp>(),
        std::any::TypeId::of::<DivOp>(),
        std::any::TypeId::of::<ExpNegOp>(),
        std::any::TypeId::of::<ExpOp>(),
        std::any::TypeId::of::<IdentityOp>(),
        std::any::TypeId::of::<LnOp>(),
        std::any::TypeId::of::<MulOp>(),
        std::any::TypeId::of::<NegOp>(),
        std::any::TypeId::of::<PowOp>(),
        std::any::TypeId::of::<RecipOp>(),
        std::any::TypeId::of::<SinOp>(),
        std::any::TypeId::of::<SqrtOp>(),
        std::any::TypeId::of::<SubOp>(),
    ];
    let unique_markers = markers.iter().collect::<std::collections::HashSet<_>>();
    assert_eq!(unique_markers.len(), markers.len());
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
fn elementwise_activation_markers_match_cpu_reference() {
    let Some(device) = device("elementwise_activation_markers_match_cpu_reference") else {
        return;
    };
    let host = [-2.0_f32, -0.5, 0.0, 0.5, 2.0];
    let input = device.upload(&host).unwrap();
    let gelu_scale = 0.797_884_6_f32;
    let gelu_cubic = 0.044715_f32;

    macro_rules! check {
        ($label:literal, $operation:ty, $expected:expr) => {{
            let expected = $expected;
            let output = unary_elementwise::<$operation, f32>(&device, &input).unwrap();
            let mut actual = [0.0_f32; 5];
            device.download(&output, &mut actual).unwrap();
            for (index, (&got, &reference)) in actual.iter().zip(expected.iter()).enumerate() {
                let tolerance = f32::EPSILON * 512.0 * reference.abs().max(1.0);
                assert!(
                    (got - reference).abs() <= tolerance,
                    "{}[{index}] got {got}, expected {reference}, tolerance {tolerance}",
                    $label
                );
            }
        }};
    }

    check!(
        "gelu_tanh",
        GeluTanhOp,
        host.iter()
            .map(|&x| 0.5 * x * (1.0 + (gelu_scale * (x + gelu_cubic * x * x * x)).tanh()))
            .collect::<Vec<_>>()
    );
    check!(
        "gelu_tanh_grad",
        GeluTanhGradOp,
        host.iter()
            .map(|&x| {
                let tanh = (gelu_scale * (x + gelu_cubic * x * x * x)).tanh();
                0.5 * (1.0 + tanh)
                    + 0.5 * x * (1.0 - tanh * tanh) * gelu_scale * (1.0 + 3.0 * gelu_cubic * x * x)
            })
            .collect::<Vec<_>>()
    );
    check!(
        "silu",
        SiluOp,
        host.iter()
            .map(|&x| x / (1.0 + (-x).exp()))
            .collect::<Vec<_>>()
    );
    check!(
        "silu_grad",
        SiluGradOp,
        host.iter()
            .map(|&x| {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 + x * (1.0 - sigmoid))
            })
            .collect::<Vec<_>>()
    );
    check!(
        "softplus",
        SoftplusOp,
        host.iter()
            .map(|&x| (1.0 + x.exp()).ln())
            .collect::<Vec<_>>()
    );
    check!(
        "softplus_grad",
        SoftplusGradOp,
        host.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect::<Vec<_>>()
    );
    check!(
        "mish",
        MishOp,
        host.iter()
            .map(|&x| x * (1.0 + x.exp()).ln().tanh())
            .collect::<Vec<_>>()
    );
    check!(
        "mish_grad",
        MishGradOp,
        host.iter()
            .map(|&x| {
                let softplus = (1.0 + x.exp()).ln();
                softplus.tanh() + x * (1.0 - softplus.tanh().powi(2)) / (1.0 + (-x).exp())
            })
            .collect::<Vec<_>>()
    );
    check!(
        "elu",
        EluOp,
        host.iter()
            .map(|&x| if x >= 0.0 { x } else { x.exp() - 1.0 })
            .collect::<Vec<_>>()
    );
    check!(
        "elu_grad",
        EluGradOp,
        host.iter()
            .map(|&x| if x >= 0.0 { 1.0 } else { x.exp() })
            .collect::<Vec<_>>()
    );
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
    let lhs_layout = Layout::try_new([2, 2], [1, 2], 1).expect("valid test layout");
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

    let probe_buffer = device
        .upload(&[1_i32, 9, 2, 0])
        .expect("HIP comparison probe upload");
    let probe_layout = Layout::c_contiguous([2, 2]).expect("comparison probe layout");
    let comparison = binary_elementwise_strided_typed::<EqOp, _, 2>(
        &device,
        lhs,
        StridedOperand {
            buffer: &probe_buffer,
            layout: &probe_layout,
        },
        [2, 2],
        width,
    )
    .expect("HIP allocated typed strided comparison");
    let mut comparison_values = [0_i32; 4];
    device
        .download(&comparison, &mut comparison_values)
        .expect("HIP typed strided comparison download");
    assert_eq!(comparison_values, [1, 0, 1, 0]);

    let allocated_scalar =
        scalar_elementwise_strided::<MulOp, _, 2>(&device, lhs, 2, [2, 2], width)
            .expect("HIP allocated strided scalar");
    let mut allocated_scalar_values = [0_i32; 4];
    device
        .download(&allocated_scalar, &mut allocated_scalar_values)
        .expect("HIP allocated strided scalar download");
    assert_eq!(allocated_scalar_values, [2, 6, 4, 8]);

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

    let aliased_layout = Layout::try_new([2, 2], [0, 1], 0).expect("valid test layout");
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
