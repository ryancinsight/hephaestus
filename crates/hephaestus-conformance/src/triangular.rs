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

    let lower0 = device.alloc_zeroed::<i32>(9).expect("lower0 alloc");
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

    let upper0 = device.alloc_zeroed::<i32>(9).expect("upper0 alloc");
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

    let lower_pos1 = device.alloc_zeroed::<i32>(9).expect("lower_pos1 alloc");
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

    let lower_neg1 = device.alloc_zeroed::<i32>(9).expect("lower_neg1 alloc");
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
        let output = device.alloc_zeroed::<i32>(9).expect("extreme alloc");
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
