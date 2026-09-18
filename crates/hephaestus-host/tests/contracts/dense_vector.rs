//! Host instantiation of the shared dense-vector conformance clause, plus the
//! host-specific aliasing and length cases the clause does not reach.

use hephaestus_conformance::assert_dense_vector_contract;
use hephaestus_core::{ComputeDevice, DenseVectorOps, HephaestusError};
use hephaestus_host::{HostDenseVectorOps, HostDevice};

fn download<const LEN: usize>(
    device: &HostDevice,
    buffer: &<HostDevice as ComputeDevice>::Buffer<f64>,
) -> [f64; LEN] {
    let mut got = [0.0f64; LEN];
    device.download(buffer, &mut got).expect("download");
    got
}

#[test]
fn host_satisfies_the_dense_vector_contract() {
    assert_dense_vector_contract(&HostDevice::new(), &HostDenseVectorOps);
}

/// `axpy(y, y, 2)` is `y += 2y`: each cell is read before it is written, so
/// `[1, 2, 3]` becomes `[3, 6, 9]`, exact in `f64`.
#[test]
fn axpy_with_source_naming_target_triples_each_cell() {
    let device = HostDevice::new();
    let y = device.upload(&[1.0f64, 2.0, 3.0]).expect("upload");
    HostDenseVectorOps
        .axpy(&device, &y, &y, 2.0)
        .expect("aliased axpy");
    assert_eq!(download::<3>(&device, &y), [3.0, 6.0, 9.0]);
}

/// `xpay(y, y, 0.5)` is `y = y + 0.5y`: `[2, 4]` becomes `[3, 6]`.
#[test]
fn xpay_with_source_naming_target_scales_each_cell() {
    let device = HostDevice::new();
    let y = device.upload(&[2.0f64, 4.0]).expect("upload");
    HostDenseVectorOps
        .xpay(&device, &y, &y, 0.5)
        .expect("aliased xpay");
    assert_eq!(download::<2>(&device, &y), [3.0, 6.0]);
}

/// `dot(a, a)` is the squared norm: `1 + 4 + 9 + 16 = 30`.
#[test]
fn dot_of_a_vector_with_itself_is_its_squared_norm() {
    let device = HostDevice::new();
    let a = device.upload(&[1.0f64, 2.0, 3.0, 4.0]).expect("upload");
    assert_eq!(HostDenseVectorOps.dot(&device, &a, &a).expect("dot"), 30.0);
}

#[test]
fn an_output_aliasing_an_input_is_rejected_before_any_write() {
    let device = HostDevice::new();
    let a = device.upload(&[1.0f64, 2.0]).expect("upload");
    let b = device.upload(&[3.0f64, 4.0]).expect("upload");
    let error = HostDenseVectorOps
        .add_into(&device, &a, &b, &a)
        .expect_err("aliased output must be rejected");
    assert!(
        matches!(error, HephaestusError::DispatchFailed { .. }),
        "{error:?}"
    );
    assert_eq!(download::<2>(&device, &a), [1.0, 2.0]);
}

#[test]
fn a_length_mismatch_is_rejected_before_any_write() {
    let device = HostDevice::new();
    let target = device.upload(&[1.0f64, 2.0, 3.0]).expect("upload");
    let source = device.upload(&[1.0f64, 1.0]).expect("upload");
    let error = HostDenseVectorOps
        .axpy(&device, &target, &source, 1.0)
        .expect_err("length mismatch must be rejected");
    assert!(
        matches!(
            error,
            HephaestusError::LengthMismatch {
                host_len: 2,
                device_len: 3
            }
        ),
        "{error:?}"
    );
    assert_eq!(download::<3>(&device, &target), [1.0, 2.0, 3.0]);
}

/// A zero-length operand is a no-op, and every norm of it is zero.
#[test]
fn zero_length_operands_are_a_no_op_with_zero_norms() {
    let device = HostDevice::new();
    let ops = HostDenseVectorOps;
    let empty = device.upload::<f64>(&[]).expect("upload");
    let other = device.upload::<f64>(&[]).expect("upload");
    ops.axpy(&device, &empty, &other, 3.0).expect("axpy");
    assert_eq!(ops.dot(&device, &empty, &other).expect("dot"), 0.0);
    assert_eq!(ops.norm_l1(&device, &empty).expect("l1"), 0.0);
    assert_eq!(ops.norm_l2(&device, &empty).expect("l2"), 0.0);
    assert_eq!(ops.norm_max(&device, &empty).expect("max"), 0.0);
}
