//! Contract clauses for the device-neutral
//! [`hephaestus_core::TriangularOps`] seam.
//!
//! Fixture `[[1,2,3],[4,5,6],[7,8,9]]` (3x3) exercises `tril`/`triu` at the
//! main diagonal (`diagonal=0`), a positive offset, and a negative offset —
//! each checked against hand-derived values.

use hephaestus_core::{ComputeDevice, StridedView, TriangularMode, TriangularOps};
use leto::Layout;

/// `tril`/`triu` at `diagonal` 0, +1, and -1, checked against hand-derived
/// values.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with the
/// expected values.
pub fn assert_triangular_contract<D, M>(device: &D, ops: &M)
where
    D: ComputeDevice,
    M: TriangularOps<D, i32>,
    D::Buffer<i32>: Clone,
{
    let name = device.backend_name();
    let layout = Layout::c_contiguous([3, 3]).expect("layout");
    let input = device
        .upload(&[1, 2, 3, 4, 5, 6, 7, 8, 9])
        .expect("fixture upload");

    let lower0 = device.upload(&[31; 9]).expect("lower0 alloc");
    ops.triangular_into(
        device,
        StridedView::new(&input, &layout),
        TriangularMode::Lower,
        0,
        StridedView::new(&lower0, &layout),
    )
    .expect("tril(0) dispatch");
    let mut got_lower0 = vec![0i32; 9];
    device.download(&lower0, &mut got_lower0).expect("download");
    assert_eq!(
        got_lower0,
        vec![1, 0, 0, 4, 5, 0, 7, 8, 9],
        "{name}: tril(diagonal=0) mismatch"
    );

    let upper0 = device.upload(&[32; 9]).expect("upper0 alloc");
    ops.triangular_into(
        device,
        StridedView::new(&input, &layout),
        TriangularMode::Upper,
        0,
        StridedView::new(&upper0, &layout),
    )
    .expect("triu(0) dispatch");
    let mut got_upper0 = vec![0i32; 9];
    device.download(&upper0, &mut got_upper0).expect("download");
    assert_eq!(
        got_upper0,
        vec![1, 2, 3, 0, 5, 6, 0, 0, 9],
        "{name}: triu(diagonal=0) mismatch"
    );

    let lower_pos1 = device.upload(&[33; 9]).expect("lower_pos1 alloc");
    ops.triangular_into(
        device,
        StridedView::new(&input, &layout),
        TriangularMode::Lower,
        1,
        StridedView::new(&lower_pos1, &layout),
    )
    .expect("tril(+1) dispatch");
    let mut got_lower_pos1 = vec![0i32; 9];
    device
        .download(&lower_pos1, &mut got_lower_pos1)
        .expect("download");
    assert_eq!(
        got_lower_pos1,
        vec![1, 2, 0, 4, 5, 6, 7, 8, 9],
        "{name}: tril(diagonal=1) mismatch"
    );

    let lower_neg1 = device.upload(&[34; 9]).expect("lower_neg1 alloc");
    ops.triangular_into(
        device,
        StridedView::new(&input, &layout),
        TriangularMode::Lower,
        -1,
        StridedView::new(&lower_neg1, &layout),
    )
    .expect("tril(-1) dispatch");
    let mut got_lower_neg1 = vec![0i32; 9];
    device
        .download(&lower_neg1, &mut got_lower_neg1)
        .expect("download");
    assert_eq!(
        got_lower_neg1,
        vec![0, 0, 0, 4, 0, 0, 7, 8, 0],
        "{name}: tril(diagonal=-1) mismatch"
    );

    let rectangular = Layout::c_contiguous([2, 3]).expect("rectangular layout");
    let rectangular_input = device
        .upload(&[10, 11, 12, 13, 14, 15])
        .expect("rectangular upload");
    let rectangular_output = device.upload(&[-7; 6]).expect("rectangular output");
    ops.triangular_into(
        device,
        StridedView::new(&rectangular_input, &rectangular),
        TriangularMode::Lower,
        0,
        StridedView::new(&rectangular_output, &rectangular),
    )
    .expect("rectangular dispatch");
    assert_eq!(
        device
            .download_owned(&rectangular_output)
            .expect("download"),
        vec![10, 0, 0, 13, 14, 0],
        "{name}: rectangular masked writes mismatch"
    );

    let reversed_columns = Layout::try_new([2, 3], [3, -1], 2).expect("negative-stride layout");
    let negative_input = device
        .upload(&[10, 11, 12, 13, 14, 15])
        .expect("negative upload");
    let negative_output = device.upload(&[-7; 6]).expect("negative output");
    ops.triangular_into(
        device,
        StridedView::new(&negative_input, &reversed_columns),
        TriangularMode::Lower,
        0,
        StridedView::new(&negative_output, &rectangular),
    )
    .expect("negative-stride dispatch");
    assert_eq!(
        device.download_owned(&negative_output).expect("download"),
        vec![12, 0, 0, 15, 14, 0],
        "{name}: negative-stride addressing mismatch"
    );

    let broadcast = Layout::try_new([2, 3], [0, 1], 0).expect("broadcast layout");
    let broadcast_input = device.upload(&[7, 8, 9]).expect("broadcast upload");
    let broadcast_output = device.upload(&[-7; 6]).expect("broadcast output");
    ops.triangular_into(
        device,
        StridedView::new(&broadcast_input, &broadcast),
        TriangularMode::Lower,
        0,
        StridedView::new(&broadcast_output, &rectangular),
    )
    .expect("broadcast dispatch");
    assert_eq!(
        device.download_owned(&broadcast_output).expect("download"),
        vec![7, 0, 0, 7, 8, 0],
        "{name}: broadcast addressing mismatch"
    );

    let overlap = Layout::try_new([2, 3], [1, 1], 0).expect("overlap layout");
    let overlap_output = device.upload(&[-9; 6]).expect("overlap output");
    let overlap_error = ops
        .triangular_into(
            device,
            StridedView::new(&rectangular_input, &rectangular),
            TriangularMode::Lower,
            0,
            StridedView::new(&overlap_output, &overlap),
        )
        .expect_err("overlapping output must be rejected");
    assert!(
        overlap_error.to_string().contains("non-overlapping"),
        "{name}: unexpected overlap error: {overlap_error}"
    );
    assert_eq!(
        device.download_owned(&overlap_output).expect("download"),
        vec![-9; 6],
        "{name}: overlap rejection mutated output"
    );

    let offset_output_layout = Layout::try_new([2, 3], [3, 1], 2).expect("offset output layout");
    let offset_output = device.upload(&[-6; 8]).expect("offset output");
    ops.triangular_into(
        device,
        StridedView::new(&rectangular_input, &rectangular),
        TriangularMode::Lower,
        0,
        StridedView::new(&offset_output, &offset_output_layout),
    )
    .expect("offset output dispatch");
    assert_eq!(
        device.download_owned(&offset_output).expect("download"),
        vec![-6, -6, 10, 0, 0, 13, 14, 0],
        "{name}: offset output addressing or sentinels mismatch"
    );

    let padded_output_layout = Layout::try_new([2, 3], [6, 2], 1).expect("padded output layout");
    let padded_output = device.upload(&[-5; 12]).expect("padded output");
    ops.triangular_into(
        device,
        StridedView::new(&rectangular_input, &rectangular),
        TriangularMode::Lower,
        0,
        StridedView::new(&padded_output, &padded_output_layout),
    )
    .expect("padded output dispatch");
    assert_eq!(
        device.download_owned(&padded_output).expect("download"),
        vec![-5, 10, -5, 0, -5, 0, -5, 13, -5, 14, -5, 0],
        "{name}: padded output addressing or backing sentinels mismatch"
    );

    let short_input = device.upload(&[1, 2, 3, 4, 5]).expect("short input");
    let full_output = device.upload(&[-8; 6]).expect("full output");
    let input_storage_error = ops
        .triangular_into(
            device,
            StridedView::new(&short_input, &rectangular),
            TriangularMode::Lower,
            0,
            StridedView::new(&full_output, &rectangular),
        )
        .expect_err("insufficient input storage must be rejected");
    assert!(
        input_storage_error.to_string().contains("storage"),
        "{name}: unexpected input storage error: {input_storage_error}"
    );
    assert_eq!(
        device.download_owned(&full_output).expect("download"),
        vec![-8; 6],
        "{name}: input storage rejection mutated output"
    );

    let full_input = device.upload(&[1, 2, 3, 4, 5, 6]).expect("full input");
    let short_output = device.upload(&[-9; 5]).expect("short output");
    let output_storage_error = ops
        .triangular_into(
            device,
            StridedView::new(&full_input, &rectangular),
            TriangularMode::Lower,
            0,
            StridedView::new(&short_output, &rectangular),
        )
        .expect_err("insufficient output storage must be rejected");
    assert!(
        output_storage_error.to_string().contains("storage"),
        "{name}: unexpected output storage error: {output_storage_error}"
    );
    assert_eq!(
        device.download_owned(&short_output).expect("download"),
        vec![-9; 5],
        "{name}: output storage rejection mutated output"
    );

    let empty = Layout::try_new([2, 0], [0, 1], 1).expect("empty layout");
    let empty_input = device.upload::<i32>(&[]).expect("empty input");
    let empty_output = device.upload(&[41, 42]).expect("empty output");
    ops.triangular_into(
        device,
        StridedView::new(&empty_input, &empty),
        TriangularMode::Lower,
        i64::MAX,
        StridedView::new(&empty_output, &empty),
    )
    .expect("empty dispatch");
    assert_eq!(
        device
            .download_owned(&empty_output)
            .expect("empty download"),
        vec![41, 42],
        "{name}: empty dispatch mutated sentinel output"
    );

    for (mode, diagonal, expected, label) in [
        (
            TriangularMode::Lower,
            i32::MAX as i64,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            "tril(i32::MAX)",
        ),
        (
            TriangularMode::Upper,
            i32::MIN as i64,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            "triu(i32::MIN)",
        ),
        (
            TriangularMode::Lower,
            i32::MIN as i64,
            vec![0; 9],
            "tril(i32::MIN)",
        ),
        (
            TriangularMode::Upper,
            i32::MAX as i64,
            vec![0; 9],
            "triu(i32::MAX)",
        ),
    ] {
        let output = device.upload(&[91; 9]).expect("extreme alloc");
        ops.triangular_into(
            device,
            StridedView::new(&input, &layout),
            mode,
            diagonal,
            StridedView::new(&output, &layout),
        )
        .expect("extreme dispatch");
        let mut got = vec![0i32; 9];
        device
            .download(&output, &mut got)
            .expect("extreme download");
        assert_eq!(got, expected, "{name}: {label} mismatch");
    }

    for (mode, diagonal, expected, label) in [
        (
            TriangularMode::Lower,
            i64::MAX,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            "tril(i64::MAX)",
        ),
        (
            TriangularMode::Upper,
            i64::MAX,
            vec![0; 9],
            "triu(i64::MAX)",
        ),
        (
            TriangularMode::Lower,
            i64::MIN,
            vec![0; 9],
            "tril(i64::MIN)",
        ),
        (
            TriangularMode::Upper,
            i64::MIN,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            "triu(i64::MIN)",
        ),
    ] {
        let output = device.upload(&[73; 9]).expect("i64 boundary output");
        ops.triangular_into(
            device,
            StridedView::new(&input, &layout),
            mode,
            diagonal,
            StridedView::new(&output, &layout),
        )
        .expect("backend accepts the full i64 diagonal contract");
        assert_eq!(
            device
                .download_owned(&output)
                .expect("i64 boundary download"),
            expected,
            "{name}: {label} mismatch"
        );
    }

    let alias = input.clone();
    let before = device
        .download_owned(&input)
        .expect("alias fixture download");
    let error = ops
        .triangular_into(
            device,
            StridedView::new(&input, &layout),
            TriangularMode::Lower,
            0,
            StridedView::new(&alias, &layout),
        )
        .expect_err("triangular dispatch must reject aliased input/output");
    assert!(
        error.to_string().contains("must not alias"),
        "{name}: unexpected alias error: {error}"
    );
    let after = device
        .download_owned(&input)
        .expect("alias fixture download");
    assert_eq!(after, before, "{name}: alias rejection mutated input");
}
