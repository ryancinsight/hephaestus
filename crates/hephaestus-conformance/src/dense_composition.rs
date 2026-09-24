//! Contract clauses for the device-neutral dense-composition role (ADR 0044).
//!
//! The fixtures use small integers and one diagonal tolerance boundary. Their
//! arithmetic is exact in `f32`, and the only non-equality comparison is the
//! documented relative-rank threshold, so the oracle is backend-neutral rather
//! than fitted to one provider's reduction order.

use hephaestus_core::{ComputeDevice, DenseCompositionOps, DeviceBuffer, StridedView};
use leto::Layout;

/// Run every dense-composition clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_dense_composition_contract<D, C>(device: &D, ops: &C)
where
    D: ComputeDevice,
    C: DenseCompositionOps<D>,
{
    matpow_matches_leto_for_dense_and_strided_inputs(device, ops);
    matpow_rejects_a_non_square_matrix(device, ops);
    determinant_matches_leto_for_regular_and_singular_matrices(device, ops);
    rank_matches_leto_and_honours_the_relative_threshold(device, ops);
}

/// Exponentiation by squaring must preserve both dense and strided inputs and
/// match Leto's integer-exact result. The empty `0 × 0` case is included
/// because it is part of the allocating wrapper's public shape contract.
fn matpow_matches_leto_for_dense_and_strided_inputs<D, C>(device: &D, ops: &C)
where
    D: ComputeDevice,
    C: DenseCompositionOps<D>,
{
    let name = device.backend_name();
    let square = Layout::c_contiguous([2, 2]).expect("2x2 layout");

    let shear_host = [1.0f32, 1.0, 0.0, 1.0];
    let shear = device.upload(&shear_host).expect("shear upload");
    let shear_power = ops
        .matpow(device, StridedView::new(&shear, &square), 5)
        .expect("dense matrix power");
    let shear_reference =
        leto::Array::from_shape_vec([2, 2], shear_host.to_vec()).expect("shear oracle");
    let expected = leto_ops::matpow(&shear_reference.view(), 5)
        .expect("leto matrix power")
        .into_vec();
    let mut actual = [0.0f32; 4];
    device
        .download(&shear_power, &mut actual)
        .expect("dense power download");
    assert_eq!(
        actual.as_slice(),
        expected.as_slice(),
        "{name}: dense matpow"
    );

    let strided_host = [99.0f32, 1.0, 2.0, 3.0, 4.0];
    let strided = device.upload(&strided_host).expect("strided upload");
    let strided_layout = Layout::try_new([2, 2], [1, 2], 1).expect("valid strided layout");
    let strided_power = ops
        .matpow(device, StridedView::new(&strided, &strided_layout), 2)
        .expect("strided matrix power");
    let mut actual = [0.0f32; 4];
    device
        .download(&strided_power, &mut actual)
        .expect("strided power download");
    assert_eq!(actual, [7.0, 15.0, 10.0, 22.0], "{name}: strided matpow");

    let identity_input_host = [99.0f32; 4];
    let identity_input = device
        .upload(&identity_input_host)
        .expect("identity-power upload");
    let identity_power = ops
        .matpow(device, StridedView::new(&identity_input, &square), 0)
        .expect("identity matrix power");
    let mut actual = [0.0f32; 4];
    device
        .download(&identity_power, &mut actual)
        .expect("identity power download");
    assert_eq!(actual, [1.0, 0.0, 0.0, 1.0], "{name}: matpow exponent zero");

    let empty = device.upload::<f32>(&[]).expect("empty upload");
    let empty_layout = Layout::c_contiguous([0, 0]).expect("empty layout");
    let empty_power = ops
        .matpow(device, StridedView::new(&empty, &empty_layout), 0)
        .expect("empty matrix power");
    assert_eq!(empty_power.len(), 0, "{name}: empty matpow output");
}

/// Non-square powers are rejected before allocation or dispatch.
fn matpow_rejects_a_non_square_matrix<D, C>(device: &D, ops: &C)
where
    D: ComputeDevice,
    C: DenseCompositionOps<D>,
{
    let host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = device.upload(&host).expect("non-square upload");
    let layout = Layout::c_contiguous([2, 3]).expect("2x3 layout");
    assert!(
        ops.matpow(device, StridedView::new(&matrix, &layout), 2)
            .is_err(),
        "{}: non-square matpow must be rejected",
        device.backend_name()
    );
}

/// Determinants are exact for these small integer matrices. The rectangular
/// operand must reject rather than treating storage length as a determinant.
fn determinant_matches_leto_for_regular_and_singular_matrices<D, C>(device: &D, ops: &C)
where
    D: ComputeDevice,
    C: DenseCompositionOps<D>,
{
    let name = device.backend_name();
    let square = Layout::c_contiguous([2, 2]).expect("2x2 layout");
    for (host, expected) in [([2.0f32, 1.0, 3.0, 4.0], 5.0), ([1.0, 2.0, 2.0, 4.0], 0.0)] {
        let matrix = device.upload(&host).expect("determinant upload");
        let output = ops
            .det(device, StridedView::new(&matrix, &square))
            .expect("determinant");
        let mut actual = [0.0f32; 1];
        device
            .download(&output, &mut actual)
            .expect("determinant download");
        let reference =
            leto::Array::from_shape_vec([2, 2], host.to_vec()).expect("determinant oracle");
        let oracle = leto_ops::det(&reference.view()).expect("leto determinant");
        assert_eq!(actual[0], expected, "{name}: exact determinant {host:?}");
        assert_eq!(actual[0], oracle, "{name}: Leto determinant {host:?}");
    }

    let strided_host = [99.0f32, 1.0, 2.0, 3.0, 4.0];
    let strided = device
        .upload(&strided_host)
        .expect("strided determinant upload");
    let strided_layout = Layout::try_new([2, 2], [1, 2], 1).expect("valid strided layout");
    let strided_determinant = ops
        .det(device, StridedView::new(&strided, &strided_layout))
        .expect("strided determinant");
    let mut actual = [0.0f32; 1];
    device
        .download(&strided_determinant, &mut actual)
        .expect("strided determinant download");
    assert_eq!(actual[0], -2.0, "{name}: strided determinant");

    let rectangular = device.alloc_zeroed::<f32>(6).expect("rectangular alloc");
    let rectangular_layout = Layout::c_contiguous([2, 3]).expect("2x3 layout");
    assert!(
        ops.det(device, StridedView::new(&rectangular, &rectangular_layout))
            .is_err(),
        "{name}: non-square determinant must be rejected"
    );
}

/// Rank uses the same default and explicit relative threshold as Leto on
/// regular fixtures, and the diagonal `diag(1, 1, 1e-4)` case makes the
/// threshold itself observable without depending on a fitted tolerance.
fn rank_matches_leto_and_honours_the_relative_threshold<D, C>(device: &D, ops: &C)
where
    D: ComputeDevice,
    C: DenseCompositionOps<D>,
{
    let name = device.backend_name();
    let square = Layout::c_contiguous([2, 2]).expect("2x2 layout");
    let full_rank_host = [1.0f32, 2.0, 3.0, 4.0];
    let full_rank = device.upload(&full_rank_host).expect("full-rank upload");
    let full_reference =
        leto::Array::from_shape_vec([2, 2], full_rank_host.to_vec()).expect("rank oracle");
    let actual = ops
        .matrix_rank(device, StridedView::new(&full_rank, &square))
        .expect("default rank");
    let expected = leto_ops::matrix_rank(&full_reference.view()).expect("leto rank");
    assert_eq!(actual, expected, "{name}: default matrix rank");

    let diagonal_host = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0e-4];
    let diagonal = device.upload(&diagonal_host).expect("diagonal rank upload");
    let diagonal_layout = Layout::c_contiguous([3, 3]).expect("3x3 layout");
    let diagonal_view = StridedView::new(&diagonal, &diagonal_layout);
    let diagonal_reference =
        leto::Array::from_shape_vec([3, 3], diagonal_host.to_vec()).expect("diagonal rank oracle");
    for (tolerance, expected) in [(1.0e-6, 3), (1.0e-2, 2)] {
        let actual = ops
            .matrix_rank_with_tolerance(device, diagonal_view, tolerance)
            .expect("tolerance rank");
        let oracle = leto_ops::matrix_rank_with_tolerance(&diagonal_reference.view(), tolerance)
            .expect("leto tolerance rank");
        assert_eq!(actual, expected, "{name}: rank at tolerance {tolerance}");
        assert_eq!(actual, oracle, "{name}: Leto rank at tolerance {tolerance}");
    }

    let deficient_host = [1.0f32, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 1.0];
    let deficient = device
        .upload(&deficient_host)
        .expect("deficient rank upload");
    let deficient_layout = Layout::c_contiguous([3, 3]).expect("3x3 layout");
    let deficient_reference = leto::Array::from_shape_vec([3, 3], deficient_host.to_vec())
        .expect("deficient rank oracle");
    let tolerance = 1.0e-6;
    let actual = ops
        .matrix_rank_with_tolerance(
            device,
            StridedView::new(&deficient, &deficient_layout),
            tolerance,
        )
        .expect("deficient rank");
    let expected = leto_ops::matrix_rank_with_tolerance(&deficient_reference.view(), tolerance)
        .expect("leto deficient rank");
    assert_eq!(actual, expected, "{name}: deficient rank");

    let zero = device.alloc_zeroed::<f32>(6).expect("zero rank alloc");
    let zero_layout = Layout::c_contiguous([2, 3]).expect("zero 2x3 layout");
    let actual = ops
        .matrix_rank(device, StridedView::new(&zero, &zero_layout))
        .expect("rectangular zero rank");
    assert_eq!(actual, 0, "{name}: rectangular zero rank");

    let strided_host = [99.0f32, 1.0, 2.0, 3.0, 4.0];
    let strided = device.upload(&strided_host).expect("strided rank upload");
    let strided_layout = Layout::try_new([2, 2], [1, 2], 1).expect("valid strided layout");
    let actual = ops
        .matrix_rank(device, StridedView::new(&strided, &strided_layout))
        .expect("strided rank");
    assert_eq!(actual, 2, "{name}: strided rank");

    let empty = device.upload::<f32>(&[]).expect("empty rank upload");
    let empty_layout = Layout::c_contiguous([0, 3]).expect("empty rank layout");
    assert!(
        ops.matrix_rank(device, StridedView::new(&empty, &empty_layout))
            .is_err(),
        "{name}: empty matrix rank must be rejected"
    );
}
