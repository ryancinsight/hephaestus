use super::*;

pub(crate) fn mixed_prepared_reduction_batch_preserves_scalar_and_axis_results() {
    use hephaestus_wgpu::StridedOperand;
    use leto::Layout;

    let Some(device) = device_or_skip() else {
        return;
    };

    let scalar_host: Vec<u32> = (0..1_024).map(|index| index % 11).collect();
    let scalar_expected: u32 = scalar_host.iter().sum();
    let scalar_input = device.upload(&scalar_host).unwrap();
    let scalar = prepare_reduction::<SumOp, u32>(&device, &scalar_input).unwrap();

    let singleton_input = device.upload(&[37u32]).unwrap();
    let singleton = prepare_reduction::<SumOp, u32>(&device, &singleton_input).unwrap();

    let empty_scalar_input = device.upload::<u32>(&[]).unwrap();
    let empty_scalar = prepare_reduction::<SumOp, u32>(&device, &empty_scalar_input).unwrap();

    let axis_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let axis_input = device.upload(&axis_host).unwrap();
    let axis_output = device.alloc_zeroed::<f32>(3).unwrap();
    let axis_input_layout = Layout::c_contiguous([2, 3]).unwrap();
    let axis_output_layout = Layout::c_contiguous([1, 3]).unwrap();
    let axis = prepare_sum_axis_into(
        &device,
        StridedOperand {
            buffer: &axis_input,
            layout: &axis_input_layout,
        },
        0,
        StridedOperand {
            buffer: &axis_output,
            layout: &axis_output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();

    submit_prepared_mixed_reduction_batch(&device, &[&scalar, &singleton, &empty_scalar], &[&axis])
        .unwrap();

    let mut scalar_actual = [0u32; 1];
    device
        .download(scalar.output(), &mut scalar_actual)
        .unwrap();
    assert_eq!(scalar_actual, [scalar_expected]);
    device
        .download(singleton.output(), &mut scalar_actual)
        .unwrap();
    assert_eq!(scalar_actual, [37]);
    device
        .download(empty_scalar.output(), &mut scalar_actual)
        .unwrap();
    assert_eq!(scalar_actual, [0]);

    let mut axis_actual = [0.0f32; 3];
    device.download(&axis_output, &mut axis_actual).unwrap();
    assert_eq!(axis_actual, [5.0, 7.0, 9.0]);

    let empty_axis_input = device.upload::<f32>(&[]).unwrap();
    let empty_axis_output = device.alloc_zeroed::<f32>(0).unwrap();
    let empty_axis_input_layout = Layout::c_contiguous([0, 3]).unwrap();
    let empty_axis_output_layout = Layout::c_contiguous([0, 1]).unwrap();
    let empty_axis = prepare_sum_axis_into(
        &device,
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
    .unwrap();
    submit_prepared_mixed_reduction_batch(&device, &[&empty_scalar], &[&empty_axis]).unwrap();
    assert_eq!(empty_axis_output.len(), 0);

    submit_prepared_mixed_reduction_batch::<u32, f32>(&device, &[], &[]).unwrap();
}

pub(crate) fn prepared_dot_reuses_output_and_observes_input_updates() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, prepare_dot};
    use leto::Layout;

    const LEN: usize = 1_025;
    let mut lhs_host = vec![1.0f32; LEN];
    let rhs_host = vec![1.0f32; LEN];
    let lhs = device.upload(&lhs_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([LEN]).unwrap();
    let prepared = prepare_dot(
        &device,
        StridedOperand {
            buffer: &lhs,
            layout: &layout,
        },
        StridedOperand {
            buffer: &rhs,
            layout: &layout,
        },
    )
    .unwrap();
    let output_handle = prepared.output().raw().clone();

    prepared.dispatch(&device).unwrap();
    let mut got = [0.0f32; 1];
    device.download(prepared.output(), &mut got).unwrap();
    assert_eq!(got, [LEN as f32]);
    assert_eq!(&output_handle, prepared.output().raw());

    lhs_host.fill(2.0);
    device.write_buffer(&lhs, &lhs_host).unwrap();
    prepared.dispatch(&device).unwrap();
    device.download(prepared.output(), &mut got).unwrap();
    assert_eq!(got, [2.0 * LEN as f32]);
    assert_eq!(&output_handle, prepared.output().raw());
}

pub(crate) fn prepared_l2_norm_reuses_output_and_observes_input_updates() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, prepare_norm_l2};
    use leto::Layout;

    const LEN: usize = 1_024;
    let mut host = vec![3.0f32; LEN];
    let input = device.upload(&host).unwrap();
    let layout = Layout::c_contiguous([LEN]).unwrap();
    let prepared = prepare_norm_l2(
        &device,
        StridedOperand {
            buffer: &input,
            layout: &layout,
        },
    )
    .unwrap();
    let output_handle = prepared.output().raw().clone();

    prepared.dispatch(&device).unwrap();
    let mut got = [0.0f32; 1];
    device.download(prepared.output(), &mut got).unwrap();
    assert_eq!(got, [96.0]);
    assert_eq!(&output_handle, prepared.output().raw());

    host.fill(4.0);
    device.write_buffer(&input, &host).unwrap();
    prepared.dispatch(&device).unwrap();
    device.download(prepared.output(), &mut got).unwrap();
    assert_eq!(got, [128.0]);
    assert_eq!(&output_handle, prepared.output().raw());
}
