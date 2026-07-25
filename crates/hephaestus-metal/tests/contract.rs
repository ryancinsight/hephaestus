//! Contract tests for the Metal `ComputeDevice` substrate and application operations.
//!
//! These run real device dispatch differentially against host references.
//! On a host without macOS or without a Metal device, [`MetalDevice::try_default`]
//! returns `Err` and each test skips. Hardware CI sets
//! `HEPHAESTUS_METAL_REQUIRE_DEVICE=1` so an unavailable device fails that lane
//! instead of being reported as device evidence.

use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError, Result};
use hephaestus_metal::{
    AddOp, MaxOp, MetalDevice, MinOp, MulOp, NegOp, SqrtOp, StridedOperand, SumOp,
    binary_elementwise, matmul, prepare_max_axis_into, prepare_mean_axis_into,
    prepare_min_axis_into, prepare_reduction, prepare_reduction_with_width, prepare_sum_axis_into,
    reduction, scalar_elementwise, submit_prepared_axis_reduction_batch,
    submit_prepared_reduction_batch, unary_elementwise, unary_elementwise_into,
};
use leto::Layout;

/// Acquire a device, or `None` to skip (no Metal device).
fn device(test: &str) -> Option<MetalDevice> {
    match MetalDevice::try_default() {
        Ok(d) => Some(d),
        Err(e) => {
            if std::env::var_os("HEPHAESTUS_METAL_REQUIRE_DEVICE").is_some() {
                panic!("Metal device required for {test}: {e}");
            }
            eprintln!("skip {test}: Metal device unavailable ({e})");
            None
        }
    }
}

fn assert_elementwise_alias_rejected(result: Result<()>) {
    match result {
        Err(HephaestusError::DispatchFailed { message }) => {
            assert!(
                message.starts_with("output buffer must not alias "),
                "unexpected alias rejection message: {message}"
            );
        }
        other => panic!("expected elementwise alias rejection, got {other:?}"),
    }
}

fn assert_length_mismatch<T>(result: Result<T>, host_len: usize, device_len: usize) {
    match result {
        Err(HephaestusError::LengthMismatch {
            host_len: got_host,
            device_len: got_device,
        }) => {
            assert_eq!(got_host, host_len);
            assert_eq!(got_device, device_len);
        }
        Err(error) => panic!("expected length mismatch {host_len}->{device_len}, got {error:?}"),
        Ok(_) => panic!("expected length mismatch {host_len}->{device_len}, got success"),
    }
}

#[test]
fn upload_download_round_trips_values() {
    let Some(d) = device("upload_download_round_trips_values") else {
        return;
    };
    let host = [1.0f32, -2.0, 3.15, 0.0];
    let buf = d.upload(&host).unwrap();
    assert_eq!(buf.len(), 4);
    let mut out = [0.0f32; 4];
    d.download(&buf, &mut out).unwrap();
    assert_eq!(out, host);
}

#[test]
fn download_rejects_length_mismatch() {
    let Some(d) = device("download_rejects_length_mismatch") else {
        return;
    };
    let buf = d.upload(&[1.0f32, 2.0]).unwrap();
    let mut out = [0.0f32; 3];
    assert_length_mismatch(d.download(&buf, &mut out), 3, 2);
}

#[test]
fn write_buffer_rejects_length_mismatch() {
    let Some(d) = device("write_buffer_rejects_length_mismatch") else {
        return;
    };
    let buf = d.upload(&[1.0f32, 2.0]).unwrap();
    let host = [1.0f32, 2.0, 3.0];
    assert_length_mismatch(d.write_buffer(&buf, &host), 3, 2);
}

#[test]
fn write_sub_buffer_overwrites_only_requested_range() {
    let Some(d) = device("write_sub_buffer_overwrites_only_requested_range") else {
        return;
    };
    let buf = d.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    d.write_sub_buffer(&buf, 1, &[20.0f32, 30.0]).unwrap();

    let mut out = [0.0f32; 4];
    d.download(&buf, &mut out).unwrap();
    assert_eq!(out, [1.0, 20.0, 30.0, 4.0]);
}

#[test]
fn write_sub_buffer_rejects_out_of_range_write() {
    let Some(d) = device("write_sub_buffer_rejects_out_of_range_write") else {
        return;
    };
    let buf = d.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    assert_length_mismatch(d.write_sub_buffer(&buf, 2, &[4.0f32, 5.0]), 4, 3);
}

#[test]
fn write_sub_buffer_empty_tail_write_is_noop() {
    let Some(d) = device("write_sub_buffer_empty_tail_write_is_noop") else {
        return;
    };
    let buf = d.upload(&[9i32, 8, 7]).unwrap();
    d.write_sub_buffer(&buf, 3, &[] as &[i32]).unwrap();

    let mut out = [0i32; 3];
    d.download(&buf, &mut out).unwrap();
    assert_eq!(out, [9, 8, 7]);
}

#[test]
fn elementwise_add_matches_cpu_reference() {
    let Some(d) = device("elementwise_add_matches_cpu_reference") else {
        return;
    };
    let a = d.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let b = d.upload(&[4.0f32, 5.0, 6.0]).unwrap();
    let out = binary_elementwise::<AddOp, f32>(&d, &a, &b).unwrap();
    let mut host_out = [0.0f32; 3];
    d.download(&out, &mut host_out).unwrap();
    assert_eq!(host_out, [5.0, 7.0, 9.0]);
}

#[test]
fn elementwise_unary_matches_cpu_reference() {
    let Some(d) = device("elementwise_unary_matches_cpu_reference") else {
        return;
    };
    let a = d.upload(&[4.0f32, 9.0, 16.0]).unwrap();
    let out = unary_elementwise::<SqrtOp, f32>(&d, &a).unwrap();
    let mut host_out = [0.0f32; 3];
    d.download(&out, &mut host_out).unwrap();
    assert_eq!(host_out, [2.0, 3.0, 4.0]);
}

#[test]
fn elementwise_scalar_matches_cpu_reference() {
    let Some(d) = device("elementwise_scalar_matches_cpu_reference") else {
        return;
    };
    let a = d.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let out = scalar_elementwise::<MulOp, f32>(&d, &a, 3.0).unwrap();
    let mut host_out = [0.0f32; 3];
    d.download(&out, &mut host_out).unwrap();
    assert_eq!(host_out, [3.0, 6.0, 9.0]);
}

#[test]
fn elementwise_into_rejects_output_input_aliasing() {
    let Some(d) = device("elementwise_into_rejects_output_input_aliasing") else {
        return;
    };
    let a = d.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    assert_elementwise_alias_rejected(unary_elementwise_into::<NegOp, f32>(
        &d,
        &a,
        &a,
        BlockWidth::DEFAULT,
    ));
}

#[test]
fn reduction_sum_matches_cpu_reference() {
    let Some(d) = device("reduction_sum_matches_cpu_reference") else {
        return;
    };
    let a = d.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let out = reduction::<SumOp, f32>(&d, &a).unwrap();
    let mut host_out = [0.0f32; 1];
    d.download(&out, &mut host_out).unwrap();
    assert_eq!(host_out[0], 10.0);
}

#[test]
fn prepared_reduction_reuses_device_outputs_and_batches() {
    let Some(d) = device("prepared_reduction_reuses_device_outputs_and_batches") else {
        return;
    };
    let input = d.upload(&[3.0f32, -2.0, 7.0, 1.0, 4.0]).unwrap();
    let width = BlockWidth::new(4).unwrap();

    let sum = prepare_reduction_with_width::<SumOp, _>(&d, &input, width).unwrap();
    sum.dispatch(&d).unwrap();
    let sum_output = sum.output();
    let mut got_sum = [0.0f32];
    d.download(&sum_output, &mut got_sum).unwrap();
    assert_eq!(got_sum, [13.0]);
    sum.dispatch(&d).unwrap();
    d.download(&sum.output(), &mut got_sum).unwrap();
    assert_eq!(got_sum, [13.0]);

    let min = prepare_reduction::<MinOp, _>(&d, &input).unwrap();
    let max = prepare_reduction::<MaxOp, _>(&d, &input).unwrap();
    submit_prepared_reduction_batch(&d, &[&min, &max]).unwrap();
    let mut got_min = [0.0f32];
    let mut got_max = [0.0f32];
    d.download(&min.output(), &mut got_min).unwrap();
    d.download(&max.output(), &mut got_max).unwrap();
    assert_eq!(got_min, [-2.0]);
    assert_eq!(got_max, [7.0]);

    let empty = d.upload::<f32>(&[]).unwrap();
    let prepared_empty = prepare_reduction::<SumOp, _>(&d, &empty).unwrap();
    prepared_empty.dispatch(&d).unwrap();
    let mut got_empty = [f32::NAN];
    d.download(&prepared_empty.output(), &mut got_empty)
        .unwrap();
    assert_eq!(got_empty, [0.0]);

    let invalid_width = BlockWidth::new(3).unwrap();
    assert!(matches!(
        prepare_reduction_with_width::<SumOp, _>(&d, &input, invalid_width),
        Err(HephaestusError::DispatchFailed { message })
            if message == "reduction block width 3 must be a power of two"
    ));
}

#[test]
fn prepared_axis_reductions_reuse_plans_and_validate_contracts() {
    let Some(d) = device("prepared_axis_reductions_reuse_plans_and_validate_contracts") else {
        return;
    };

    let host: Vec<f32> = (1..=12).map(|value| value as f32).collect();
    let input = d.upload(&host).unwrap();
    let input_layout = Layout::c_contiguous([3, 4]).unwrap();
    let input_operand = StridedOperand {
        buffer: &input,
        layout: &input_layout,
    };
    let width = BlockWidth::new(2).unwrap();

    let axis0_out = d.alloc_zeroed::<f32>(4).unwrap();
    let axis0_layout = Layout::c_contiguous([1, 4]).unwrap();
    let prepared_sum_axis0 = prepare_sum_axis_into(
        &d,
        input_operand,
        0,
        StridedOperand {
            buffer: &axis0_out,
            layout: &axis0_layout,
        },
        width,
    )
    .unwrap();
    prepared_sum_axis0.dispatch(&d).unwrap();
    let mut got_axis0 = [0.0f32; 4];
    d.download(&axis0_out, &mut got_axis0).unwrap();
    assert_eq!(got_axis0, [15.0, 18.0, 21.0, 24.0]);
    prepared_sum_axis0.dispatch(&d).unwrap();
    d.download(&axis0_out, &mut got_axis0).unwrap();
    assert_eq!(got_axis0, [15.0, 18.0, 21.0, 24.0]);

    let transposed_layout = Layout::new([4, 3], [1, 4], 0);
    let transposed_input = StridedOperand {
        buffer: &input,
        layout: &transposed_layout,
    };
    let axis1_out = d.alloc_zeroed::<f32>(4).unwrap();
    let axis1_layout = Layout::c_contiguous([4, 1]).unwrap();
    let prepared_sum_axis1 = prepare_sum_axis_into(
        &d,
        transposed_input,
        1,
        StridedOperand {
            buffer: &axis1_out,
            layout: &axis1_layout,
        },
        width,
    )
    .unwrap();
    let max_axis0_out = d.alloc_zeroed::<f32>(3).unwrap();
    let max_axis0_layout = Layout::c_contiguous([1, 3]).unwrap();
    let prepared_max_axis0 = prepare_max_axis_into(
        &d,
        transposed_input,
        0,
        StridedOperand {
            buffer: &max_axis0_out,
            layout: &max_axis0_layout,
        },
        width,
    )
    .unwrap();
    submit_prepared_axis_reduction_batch(&d, &[&prepared_sum_axis1, &prepared_max_axis0]).unwrap();
    let mut got_axis1 = [0.0f32; 4];
    let mut got_max_axis0 = [0.0f32; 3];
    d.download(&axis1_out, &mut got_axis1).unwrap();
    d.download(&max_axis0_out, &mut got_max_axis0).unwrap();
    assert_eq!(got_axis1, [15.0, 18.0, 21.0, 24.0]);
    assert_eq!(got_max_axis0, [9.0, 10.0, 11.0]);

    let mean_axis1_out = d.alloc_zeroed::<f32>(4).unwrap();
    let prepared_mean_axis1 = prepare_mean_axis_into(
        &d,
        transposed_input,
        1,
        StridedOperand {
            buffer: &mean_axis1_out,
            layout: &axis1_layout,
        },
        width,
    )
    .unwrap();
    prepared_mean_axis1.dispatch(&d).unwrap();
    let mut got_mean_axis1 = [0.0f32; 4];
    d.download(&mean_axis1_out, &mut got_mean_axis1).unwrap();
    assert_eq!(got_mean_axis1, [5.0, 6.0, 7.0, 8.0]);

    let empty_input = d.upload::<f32>(&[]).unwrap();
    let empty_input_layout = Layout::c_contiguous([3, 0]).unwrap();
    let empty_output = d.upload(&[7.0f32; 3]).unwrap();
    let empty_output_layout = Layout::c_contiguous([3, 1]).unwrap();
    let prepared_empty_sum = prepare_sum_axis_into(
        &d,
        StridedOperand {
            buffer: &empty_input,
            layout: &empty_input_layout,
        },
        1,
        StridedOperand {
            buffer: &empty_output,
            layout: &empty_output_layout,
        },
        width,
    )
    .unwrap();
    prepared_empty_sum.dispatch(&d).unwrap();
    let mut got_empty = [7.0f32; 3];
    d.download(&empty_output, &mut got_empty).unwrap();
    assert_eq!(got_empty, [0.0, 0.0, 0.0]);

    let empty_min = prepare_min_axis_into(
        &d,
        StridedOperand {
            buffer: &empty_input,
            layout: &empty_input_layout,
        },
        1,
        StridedOperand {
            buffer: &empty_output,
            layout: &empty_output_layout,
        },
        width,
    );
    assert!(matches!(
        empty_min,
        Err(HephaestusError::DispatchFailed { message })
            if message == "min_axis is undefined for empty axis 1"
    ));

    let alias_layout = Layout::c_contiguous([3, 1]).unwrap();
    let alias = prepare_sum_axis_into(
        &d,
        input_operand,
        1,
        StridedOperand {
            buffer: &input,
            layout: &alias_layout,
        },
        width,
    );
    assert!(matches!(
        alias,
        Err(HephaestusError::DispatchFailed { message })
            if message == "axis reduction output buffer must not alias input buffer"
    ));

    let invalid_width = BlockWidth::new(3).unwrap();
    let invalid = prepare_sum_axis_into(
        &d,
        input_operand,
        0,
        StridedOperand {
            buffer: &axis0_out,
            layout: &axis0_layout,
        },
        invalid_width,
    );
    assert!(matches!(
        invalid,
        Err(HephaestusError::DispatchFailed { message })
            if message == "reduction block width 3 must be a power of two"
    ));
}

#[test]
fn linalg_matmul_matches_cpu_reference() {
    let Some(d) = device("linalg_matmul_matches_cpu_reference") else {
        return;
    };
    let a = d.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let b = d.upload(&[5.0f32, 6.0, 7.0, 8.0]).unwrap();
    let out = matmul(
        &d,
        StridedOperand {
            buffer: &a,
            layout: &Layout::c_contiguous([2, 2]).unwrap(),
        },
        StridedOperand {
            buffer: &b,
            layout: &Layout::c_contiguous([2, 2]).unwrap(),
        },
    )
    .unwrap();
    let mut host_out = [0.0f32; 4];
    d.download(&out, &mut host_out).unwrap();
    assert_eq!(host_out, [19.0, 22.0, 43.0, 50.0,]);
}
