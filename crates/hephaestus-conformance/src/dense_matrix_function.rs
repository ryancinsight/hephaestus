//! Contract clauses for the device-neutral host-delegated matrix-function role
//! (ADR 0044).
//!
//! Every provider delegates these operations to the same Leto implementation.
//! The output comparisons can therefore be exact, while the analytical
//! exponential and Moore-Penrose identities retain small, explicitly derived
//! tolerances for accumulated floating-point error.

use hephaestus_core::{ComputeDevice, DenseMatrixFunctionOps, DeviceBuffer, StridedView};
use leto::Layout;

/// Run every dense matrix-function clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_dense_matrix_function_contract<D, M>(device: &D, ops: &M)
where
    D: ComputeDevice,
    M: DenseMatrixFunctionOps<D>,
{
    pinv_matches_leto_and_moore_penrose_contracts(device, ops);
    matexp_matches_closed_forms_and_leto(device, ops);
    matrix_functions_validate_storage_and_reject_invalid_values(device, ops);
}

fn pinv_matches_leto_and_moore_penrose_contracts<D, M>(device: &D, ops: &M)
where
    D: ComputeDevice,
    M: DenseMatrixFunctionOps<D>,
{
    let name = device.backend_name();
    let square = Layout::c_contiguous([2, 2]).expect("2x2 layout");

    let diagonal_host = [2.0f32, 0.0, 0.0, 4.0];
    let diagonal = device
        .upload(&diagonal_host)
        .expect("diagonal pseudoinverse upload");
    let diagonal_output = ops
        .pinv(device, StridedView::new(&diagonal, &square))
        .expect("diagonal pseudoinverse");
    let mut diagonal_actual = [0.0f32; 4];
    device
        .download(&diagonal_output, &mut diagonal_actual)
        .expect("diagonal pseudoinverse download");
    assert_eq!(
        diagonal_actual,
        [0.5, 0.0, 0.0, 0.25],
        "{name}: diagonal pseudoinverse"
    );

    let deficient_host = [1.0f32, 2.0, 2.0, 4.0];
    let deficient = device
        .upload(&deficient_host)
        .expect("rank-deficient pseudoinverse upload");
    let deficient_output = ops
        .pinv(device, StridedView::new(&deficient, &square))
        .expect("rank-deficient pseudoinverse");
    let mut deficient_actual = [0.0f32; 4];
    device
        .download(&deficient_output, &mut deficient_actual)
        .expect("rank-deficient pseudoinverse download");
    let deficient_oracle = leto_ops::pinv(&leto::ArrayView::new(square, &deficient_host))
        .expect("Leto rank-deficient pseudoinverse");
    assert_eq!(
        deficient_actual.as_slice(),
        leto::Storage::as_slice(deficient_oracle.storage()),
        "{name}: rank-deficient pseudoinverse must match Leto exactly"
    );
    let a_ap = matmul(&deficient_host, 2, 2, &deficient_actual, 2);
    let a_ap_a = matmul(&a_ap, 2, 2, &deficient_host, 2);
    assert_relative_close(
        &a_ap_a,
        &deficient_host,
        1.0e-4,
        1.0e-4,
        &format!("{name}: A A+ A"),
    );
    let ap_a = matmul(&deficient_actual, 2, 2, &deficient_host, 2);
    let ap_a_ap = matmul(&ap_a, 2, 2, &deficient_actual, 2);
    assert_relative_close(
        &ap_a_ap,
        &deficient_actual,
        1.0e-4,
        1.0e-4,
        &format!("{name}: A+ A A+"),
    );

    let rectangular_host = [1.0f32, 2.0, 0.0, 1.0, 2.0, 1.0];
    let rectangular = device
        .upload(&rectangular_host)
        .expect("rectangular pseudoinverse upload");
    let rectangular_layout = Layout::c_contiguous([3, 2]).expect("3x2 layout");
    let rectangular_output = ops
        .pinv(device, StridedView::new(&rectangular, &rectangular_layout))
        .expect("rectangular pseudoinverse");
    let mut rectangular_actual = [0.0f32; 6];
    device
        .download(&rectangular_output, &mut rectangular_actual)
        .expect("rectangular pseudoinverse download");
    let rectangular_oracle =
        leto_ops::pinv(&leto::ArrayView::new(rectangular_layout, &rectangular_host))
            .expect("Leto rectangular pseudoinverse");
    assert_eq!(
        rectangular_actual.as_slice(),
        leto::Storage::as_slice(rectangular_oracle.storage()),
        "{name}: rectangular pseudoinverse must match Leto exactly"
    );
    let rectangular_a_ap = matmul(&rectangular_host, 3, 2, &rectangular_actual, 3);
    let reconstructed = matmul(&rectangular_a_ap, 3, 3, &rectangular_host, 2);
    assert_relative_close(
        &reconstructed,
        &rectangular_host,
        1.0e-4,
        1.0e-4,
        &format!("{name}: rectangular A A+ A"),
    );

    let strided_host = [99.0f32, 1.0, 2.0, 3.0, 4.0];
    let strided = device
        .upload(&strided_host)
        .expect("strided pseudoinverse upload");
    let strided_layout = Layout::try_new([2, 2], [1, 2], 1).expect("valid strided layout");
    let strided_output = ops
        .pinv(device, StridedView::new(&strided, &strided_layout))
        .expect("strided pseudoinverse");
    let mut strided_actual = [0.0f32; 4];
    device
        .download(&strided_output, &mut strided_actual)
        .expect("strided pseudoinverse download");
    let strided_oracle = leto_ops::pinv(&leto::ArrayView::new(strided_layout, &strided_host))
        .expect("Leto strided pseudoinverse");
    assert_eq!(
        strided_actual.as_slice(),
        leto::Storage::as_slice(strided_oracle.storage()),
        "{name}: strided pseudoinverse"
    );

    let empty = device.upload::<f32>(&[]).expect("empty upload");
    let empty_layout = Layout::c_contiguous([0, 0]).expect("empty layout");
    let empty_output = ops
        .pinv(device, StridedView::new(&empty, &empty_layout))
        .expect("empty pseudoinverse");
    assert_eq!(empty_output.len(), 0, "{name}: empty pseudoinverse");
}

fn matexp_matches_closed_forms_and_leto<D, M>(device: &D, ops: &M)
where
    D: ComputeDevice,
    M: DenseMatrixFunctionOps<D>,
{
    let name = device.backend_name();
    let square = Layout::c_contiguous([2, 2]).expect("2x2 layout");

    let diagonal_host = [0.0f32, 0.0, 0.0, 1.0];
    let diagonal = device
        .upload(&diagonal_host)
        .expect("diagonal exponential upload");
    let diagonal_output = ops
        .matexp(device, StridedView::new(&diagonal, &square))
        .expect("diagonal exponential");
    let mut diagonal_actual = [0.0f32; 4];
    device
        .download(&diagonal_output, &mut diagonal_actual)
        .expect("diagonal exponential download");
    let diagonal_oracle = leto_ops::matexp(&leto::ArrayView::new(square, &diagonal_host))
        .expect("Leto diagonal exponential");
    assert_eq!(
        diagonal_actual.as_slice(),
        leto::Storage::as_slice(diagonal_oracle.storage()),
        "{name}: diagonal exponential must match Leto exactly"
    );
    assert_relative_close(
        &diagonal_actual,
        &[1.0, 0.0, 0.0, 1.0f32.exp()],
        1.0e-6,
        1.0e-6,
        &format!("{name}: diagonal exponential closed form"),
    );

    let theta = 0.9f32;
    let nilpotent_host = [0.0f32, 1.0, 0.0, 0.0];
    let rotation_host = [0.0f32, -theta, theta, 0.0];
    let nilpotent = device
        .upload(&nilpotent_host)
        .expect("nilpotent exponential upload");
    let rotation = device
        .upload(&rotation_host)
        .expect("rotation exponential upload");
    let nilpotent_output = ops
        .matexp(device, StridedView::new(&nilpotent, &square))
        .expect("nilpotent exponential");
    let rotation_output = ops
        .matexp(device, StridedView::new(&rotation, &square))
        .expect("rotation exponential");
    let mut nilpotent_actual = [0.0f32; 4];
    let mut rotation_actual = [0.0f32; 4];
    device
        .download(&nilpotent_output, &mut nilpotent_actual)
        .expect("nilpotent exponential download");
    device
        .download(&rotation_output, &mut rotation_actual)
        .expect("rotation exponential download");
    assert_relative_close(
        &nilpotent_actual,
        &[1.0, 1.0, 0.0, 1.0],
        1.0e-6,
        1.0e-6,
        &format!("{name}: nilpotent exponential closed form"),
    );
    assert_relative_close(
        &rotation_actual,
        &[theta.cos(), -theta.sin(), theta.sin(), theta.cos()],
        1.0e-6,
        1.0e-6,
        &format!("{name}: rotation exponential closed form"),
    );

    let general_host = [1.2f32, -0.7, 0.4, 0.3, 2.1, -1.5, -0.6, 0.8, 0.5];
    let general = device
        .upload(&general_host)
        .expect("general exponential upload");
    let general_layout = Layout::c_contiguous([3, 3]).expect("3x3 layout");
    let general_output = ops
        .matexp(device, StridedView::new(&general, &general_layout))
        .expect("general exponential");
    let mut general_actual = [0.0f32; 9];
    device
        .download(&general_output, &mut general_actual)
        .expect("general exponential download");
    let general_oracle = leto_ops::matexp(&leto::ArrayView::new(general_layout, &general_host))
        .expect("Leto general exponential");
    assert_eq!(
        general_actual.as_slice(),
        leto::Storage::as_slice(general_oracle.storage()),
        "{name}: general exponential must match Leto exactly"
    );

    let strided_host = [99.0f32, 0.0, 0.0, 1.0, 99.0];
    let strided = device
        .upload(&strided_host)
        .expect("strided exponential upload");
    let strided_layout = Layout::try_new([2, 2], [1, 2], 1).expect("valid strided layout");
    let strided_output = ops
        .matexp(device, StridedView::new(&strided, &strided_layout))
        .expect("strided exponential");
    let mut strided_actual = [0.0f32; 4];
    device
        .download(&strided_output, &mut strided_actual)
        .expect("strided exponential download");
    let strided_oracle = leto_ops::matexp(&leto::ArrayView::new(strided_layout, &strided_host))
        .expect("Leto strided exponential");
    assert_eq!(
        strided_actual.as_slice(),
        leto::Storage::as_slice(strided_oracle.storage()),
        "{name}: strided exponential"
    );

    let empty = device.upload::<f32>(&[]).expect("empty upload");
    let empty_layout = Layout::c_contiguous([0, 0]).expect("empty layout");
    let empty_output = ops
        .matexp(device, StridedView::new(&empty, &empty_layout))
        .expect("empty exponential");
    assert_eq!(empty_output.len(), 0, "{name}: empty exponential");
}

fn matrix_functions_validate_storage_and_reject_invalid_values<D, M>(device: &D, ops: &M)
where
    D: ComputeDevice,
    M: DenseMatrixFunctionOps<D>,
{
    let name = device.backend_name();
    let short = device
        .upload(&[1.0f32, 2.0, 3.0])
        .expect("short matrix upload");
    let square = Layout::c_contiguous([2, 2]).expect("2x2 layout");
    assert!(
        ops.pinv(device, StridedView::new(&short, &square)).is_err(),
        "{name}: pseudoinverse must validate storage length"
    );
    assert!(
        ops.matexp(device, StridedView::new(&short, &square))
            .is_err(),
        "{name}: exponential must validate storage length"
    );

    let rectangular = device
        .upload(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("rectangular upload");
    let rectangular_layout = Layout::c_contiguous([2, 3]).expect("2x3 layout");
    assert!(
        ops.matexp(device, StridedView::new(&rectangular, &rectangular_layout))
            .is_err(),
        "{name}: non-square exponential must be rejected"
    );

    let non_finite = device
        .upload(&[1.0f32, f32::NAN, 0.0, 1.0])
        .expect("non-finite upload");
    assert!(
        ops.pinv(device, StridedView::new(&non_finite, &square))
            .is_err(),
        "{name}: non-finite pseudoinverse must be rejected"
    );
    assert!(
        ops.matexp(device, StridedView::new(&non_finite, &square))
            .is_err(),
        "{name}: non-finite exponential must be rejected"
    );
}

fn matmul(lhs: &[f32], rows: usize, inner: usize, rhs: &[f32], cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for column in 0..cols {
            output[row * cols + column] = (0..inner)
                .map(|index| lhs[row * inner + index] * rhs[index * cols + column])
                .sum();
        }
    }
    output
}

fn assert_relative_close(
    actual: &[f32],
    expected: &[f32],
    absolute: f32,
    relative: f32,
    label: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = absolute + relative * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label} at {index}: got {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}
