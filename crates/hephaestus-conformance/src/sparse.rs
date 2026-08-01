//! Contract clauses for the device-neutral sparse operator seam.
//!
//! Each clause is generic over the device and the seam, so every backend runs
//! the same assertions. The fixture is the 3x3 CSR matrix
//! `[[2,0,1],[0,3,0],[4,0,5]]` applied to `x = [1,2,3]`: every product and
//! sum is a small integer, so `y = [5,6,19]` is exact in `f32` and every
//! oracle is an exact equality. Structural rejection clauses cover each
//! documented CSR invariant, and a rejected upload or apply must leave
//! nothing partially mutated.

use hephaestus_core::{ComputeDevice, SparseOperatorOps};

/// The 3x3 fixture's CSR parts.
fn fixture() -> (Vec<f32>, Vec<usize>, Vec<usize>) {
    (
        vec![2.0, 1.0, 3.0, 4.0, 5.0],
        vec![0, 2, 1, 0, 2],
        vec![0, 2, 3, 5],
    )
}

/// Run every sparse-operator clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_sparse_operator_contract<D, S>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SparseOperatorOps<D, f32>,
{
    let name = device.backend_name();
    let (values, cols, row_ptr) = fixture();

    // Upload and shape.
    let matrix = ops
        .upload_csr(device, &values, &cols, &row_ptr, 3, 3)
        .expect("fixture upload");
    assert_eq!(ops.shape(&matrix), (3, 3), "{name}: CSR shape");

    // Exact SpMV: [[2,0,1],[0,3,0],[4,0,5]] · [1,2,3] = [5,6,19].
    let x = device.upload(&[1.0f32, 2.0, 3.0]).expect("input upload");
    let mut y = device.alloc_zeroed::<f32>(3).expect("output alloc");
    ops.apply(device, &matrix, &x, &mut y).expect("spmv");
    let mut got = [0.0f32; 3];
    device.download(&y, &mut got).expect("download");
    assert_eq!(got, [5.0, 6.0, 19.0], "{name}: exact SpMV");

    // Re-application over new input contents is exact, not cached:
    // A · [2,0,1] = [5,0,13].
    device
        .write_buffer(&x, &[2.0f32, 0.0, 1.0])
        .expect("input rewrite");
    ops.apply(device, &matrix, &x, &mut y).expect("second spmv");
    device.download(&y, &mut got).expect("download");
    assert_eq!(
        got,
        [5.0, 0.0, 13.0],
        "{name}: SpMV must read current input contents"
    );

    // Structural rejections: each documented CSR invariant, none mutating.
    struct Malformed {
        label: &'static str,
        values: Vec<f32>,
        cols: Vec<usize>,
        row_ptr: Vec<usize>,
    }
    let cases = [
        Malformed {
            label: "short row_ptr",
            values: values.clone(),
            cols: cols.clone(),
            row_ptr: vec![0, 2, 5],
        },
        Malformed {
            label: "decreasing row_ptr",
            values: values.clone(),
            cols: cols.clone(),
            row_ptr: vec![0, 3, 2, 5],
        },
        Malformed {
            label: "unsorted columns in a row",
            values: values.clone(),
            cols: vec![2, 0, 1, 0, 2],
            row_ptr: row_ptr.clone(),
        },
        Malformed {
            label: "column index out of range",
            values: values.clone(),
            cols: vec![0, 3, 1, 0, 2],
            row_ptr: row_ptr.clone(),
        },
    ];
    for case in cases {
        assert!(
            ops.upload_csr(device, &case.values, &case.cols, &case.row_ptr, 3, 3)
                .is_err(),
            "{name}: upload must reject {}",
            case.label
        );
    }
    assert!(
        ops.upload_csr(device, &values, &cols[..4], &row_ptr, 3, 3)
            .is_err(),
        "{name}: upload must reject a values/col_indices length mismatch"
    );
}
