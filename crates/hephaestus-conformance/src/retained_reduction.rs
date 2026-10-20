//! Contract clauses for the owned dense-reduction seam.
//!
//! Each clause is generic over the device, the seam, and the fixture scalar,
//! so every backend runs the same assertions per scalar. Fixtures reuse the
//! dense-vector clause's exactly representable values: integer products and
//! sums compare exactly, while norms permit one output ULP for the backend
//! square-root operation.
//!
//! Beyond values, the clauses pin what makes a reduction *retained*: each
//! handle moves into a workspace struct beside the vectors it measures —
//! the shape the borrowing `prepare_dot` form cannot express — and observes
//! writes made through that workspace.

use hephaestus_core::{ComputeDevice, DenseVectorOps, RetainedReductions};

use super::dense_vector::{ContractScalar, assert_one_ulp, fixture};

/// Run every retained-reduction clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_retained_reduction_contract<D, E, T>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: DenseVectorOps<D, T> + RetainedReductions<D, T>,
    T: ContractScalar,
{
    let name = device.backend_name();

    // Retained dot: matches the one-shot value, observes writes through a
    // workspace-owned handle, and rejects foreign operands.
    let a = device
        .upload(&fixture([1.0, 2.0, 3.0, 4.0]))
        .expect("dot lhs");
    let b = device
        .upload(&fixture([8.0, 4.0, 2.0, 1.0]))
        .expect("dot rhs");
    assert_eq!(
        ops.dot(device, &a, &b).expect("one-shot dot"),
        T::from_fixture(26.0),
        "{name}: one-shot dot reference"
    );
    let retained = ops.retain_dot(device, &a, &b).expect("retain dot");
    assert_eq!(
        ops.dot_retained(device, &retained, &a, &b).expect("dot"),
        T::from_fixture(26.0),
        "{name}: retained dot"
    );
    struct RetainedDotWorkspace<B, R> {
        left: B,
        right: B,
        dot: R,
    }
    let workspace = RetainedDotWorkspace {
        left: a,
        right: b,
        dot: retained,
    };
    // In-place scale through the workspace: [1,2,3,4] * 2 = [2,4,6,8], and
    // 2*8 + 4*4 + 6*2 + 8*1 = 52, exact.
    ops.scale_vector(device, &workspace.left, T::from_fixture(2.0))
        .expect("scale");
    assert_eq!(
        ops.dot_retained(device, &workspace.dot, &workspace.left, &workspace.right)
            .expect("dot"),
        T::from_fixture(52.0),
        "{name}: retained dot must observe writes to its bound operand"
    );
    let foreign = device
        .upload(&fixture([1.0, 1.0, 1.0, 1.0]))
        .expect("foreign");
    assert!(
        ops.dot_retained(device, &workspace.dot, &foreign, &workspace.right)
            .is_err(),
        "{name}: retained dot must reject an operand it was not retained against"
    );
    let short = device.upload(&fixture([1.0, 1.0])).expect("short");
    assert!(
        ops.dot_retained(device, &workspace.dot, &short, &workspace.right)
            .is_err(),
        "{name}: retained dot must reject a differently-sized operand"
    );

    // Retained norm: the Pythagorean quadruple [2,3,6,0] has norm 7. The
    // integer sum of squares is exact; only the backend sqrt may round.
    let v = device
        .upload(&fixture([2.0, 3.0, 6.0, 0.0]))
        .expect("norm vector");
    assert_one_ulp::<T>(
        ops.norm_l2(device, &v).expect("one-shot norm"),
        7.0,
        name,
        "one-shot norm reference",
    );
    let retained = ops.retain_norm_l2(device, &v).expect("retain norm");
    assert_one_ulp::<T>(
        ops.norm_l2_retained(device, &retained, &v).expect("norm"),
        7.0,
        name,
        "retained norm",
    );
    struct RetainedNormWorkspace<B, R> {
        vector: B,
        norm: R,
    }
    let workspace = RetainedNormWorkspace {
        vector: v,
        norm: retained,
    };
    // Rebind: [9,12,20,0] is a Pythagorean quadruple with norm exactly 25.
    device
        .write_buffer(&workspace.vector, &fixture([9.0, 12.0, 20.0, 0.0]))
        .expect("vector rewrite");
    assert_one_ulp::<T>(
        ops.norm_l2_retained(device, &workspace.norm, &workspace.vector)
            .expect("norm"),
        25.0,
        name,
        "retained norm must observe writes to its bound operand",
    );
    let foreign = device
        .upload(&fixture([1.0, 1.0, 1.0, 1.0]))
        .expect("foreign");
    assert!(
        ops.norm_l2_retained(device, &workspace.norm, &foreign)
            .is_err(),
        "{name}: retained norm must reject an operand it was not retained against"
    );
}
