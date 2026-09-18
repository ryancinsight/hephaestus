//! Host instantiation of the shared sparse-operator and batch-submit
//! clauses, plus the host-specific aliasing and length cases.

use hephaestus_conformance::{assert_batch_submit_contract, assert_sparse_operator_contract};
use hephaestus_core::{ComputeDevice, HephaestusError, SparseOperatorOps};
use hephaestus_host::{HostDevice, HostSparseOps};

#[test]
fn host_satisfies_the_sparse_operator_contract() {
    assert_sparse_operator_contract(&HostDevice::new(), &HostSparseOps);
}

#[test]
fn host_satisfies_the_batch_submit_contract() {
    assert_batch_submit_contract(&HostDevice::new(), &HostSparseOps);
}

/// The 2x2 identity in CSR.
fn identity(device: &HostDevice) -> leto_ops::CsrMatrix<f64> {
    HostSparseOps
        .upload_csr(device, &[1.0, 1.0], &[0, 1], &[0, 1, 2], 2, 2)
        .expect("identity upload")
}

#[test]
fn an_output_aliasing_the_input_is_rejected() {
    let device = HostDevice::new();
    let matrix = identity(&device);
    let x = device.upload(&[1.0f64, 2.0]).expect("upload");
    let mut same = x.clone();
    let error = HostSparseOps
        .apply(&device, &matrix, &x, &mut same)
        .expect_err("aliased output must be rejected");
    assert!(
        matches!(error, HephaestusError::DispatchFailed { .. }),
        "{error:?}"
    );
}

#[test]
fn an_input_shorter_than_the_matrix_columns_is_rejected_before_any_write() {
    let device = HostDevice::new();
    let matrix = identity(&device);
    let x = device.upload(&[1.0f64]).expect("upload");
    let mut y = device.upload(&[7.0f64, 7.0]).expect("upload");
    let error = HostSparseOps
        .apply(&device, &matrix, &x, &mut y)
        .expect_err("short input must be rejected");
    assert!(
        matches!(
            error,
            HephaestusError::LengthMismatch {
                host_len: 1,
                device_len: 2
            }
        ),
        "{error:?}"
    );
    let mut got = [0.0f64; 2];
    device.download(&y, &mut got).expect("download");
    assert_eq!(got, [7.0, 7.0]);
}

/// A CSR row count with no room for `row_ptr`'s extra entry is rejected,
/// not an overflow panic.
#[test]
fn a_row_count_whose_row_ptr_length_overflows_is_rejected() {
    let device = HostDevice::new();
    let result = HostSparseOps.upload_csr(&device, &[] as &[f64], &[], &[0], usize::MAX, 1);
    assert!(
        matches!(result, Err(HephaestusError::DispatchFailed { .. })),
        "upload must reject rows = usize::MAX"
    );
}
