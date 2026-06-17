//! Contract tests for the Metal `ComputeDevice` substrate and application operations.
//!
//! These run real device dispatch differentially against host references.
//! On a host without macOS or without a Metal device, [`MetalDevice::try_default`]
//! returns `Err` and each test skips.

use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError, Result};
use hephaestus_metal::{
    binary_elementwise, matmul, reduction, scalar_elementwise, unary_elementwise,
    unary_elementwise_into, AddOp, MetalDevice, MulOp, NegOp, SqrtOp, StridedOperand, SumOp,
};
use leto::Layout;

/// Acquire a device, or `None` to skip (no Metal device).
fn device(test: &str) -> Option<MetalDevice> {
    match MetalDevice::try_default() {
        Ok(d) => Some(d),
        Err(e) => {
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
