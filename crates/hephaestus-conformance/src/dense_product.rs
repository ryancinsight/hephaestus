//! Contract clauses for the device-neutral dense product seam (ADR 0044).
//!
//! Every fixture is a small integer matrix, so all products are exact in
//! `f32` and every oracle is an exact equality.

use hephaestus_core::{ComputeDevice, DenseProductOps, StridedView};
use leto::Layout;

/// Run every dense-product clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_dense_product_contract<D, P>(device: &D, ops: &P)
where
    D: ComputeDevice,
    P: DenseProductOps<D, f32>,
{
    matmul_computes_the_exact_product(device, ops);
    matmul_reads_through_strided_views(device, ops);
    matmul_rejects_a_shared_dimension_mismatch(device, ops);
    batched_matmul_computes_per_batch_products(device, ops);
    kron_computes_the_exact_tensor_product(device, ops);
}

/// `[[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]`, exactly.
fn matmul_computes_the_exact_product<D, P>(device: &D, ops: &P)
where
    D: ComputeDevice,
    P: DenseProductOps<D, f32>,
{
    let name = device.backend_name();
    let lhs = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).expect("lhs upload");
    let rhs = device.upload(&[5.0f32, 6.0, 7.0, 8.0]).expect("rhs upload");
    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    let two = Layout::c_contiguous([2, 2]).expect("2x2 layout");
    ops.matmul_into(
        device,
        StridedView::new(&lhs, &two),
        StridedView::new(&rhs, &two),
        StridedView::new(&out, &two),
    )
    .expect("matmul dispatch");
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(got, [19.0, 22.0, 43.0, 50.0], "{name}: exact matmul");
}

/// A transposed lhs view multiplies as the transpose: `Aᵀ·B` differs from
/// `A·B` and matches the host oracle.
fn matmul_reads_through_strided_views<D, P>(device: &D, ops: &P)
where
    D: ComputeDevice,
    P: DenseProductOps<D, f32>,
{
    let name = device.backend_name();
    let lhs = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).expect("lhs upload");
    let rhs = device.upload(&[5.0f32, 6.0, 7.0, 8.0]).expect("rhs upload");
    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    let two = Layout::c_contiguous([2, 2]).expect("2x2 layout");
    // Aᵀ = [[1,3],[2,4]]; Aᵀ·B = [[26,30],[38,44]].
    let transposed = Layout::new([2, 2], [1, 2], 0);
    ops.matmul_into(
        device,
        StridedView::new(&lhs, &transposed),
        StridedView::new(&rhs, &two),
        StridedView::new(&out, &two),
    )
    .expect("transposed matmul dispatch");
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [26.0, 30.0, 38.0, 44.0],
        "{name}: matmul must traverse the transposed lhs layout"
    );
}

/// A 2x3 · 2x2 product is rejected before any output element is written.
fn matmul_rejects_a_shared_dimension_mismatch<D, P>(device: &D, ops: &P)
where
    D: ComputeDevice,
    P: DenseProductOps<D, f32>,
{
    let name = device.backend_name();
    let lhs = device
        .upload(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("lhs upload");
    let rhs = device.upload(&[5.0f32, 6.0, 7.0, 8.0]).expect("rhs upload");
    let out = device.upload(&[9.0f32, 9.0, 9.0, 9.0]).expect("sentinel");
    let lhs_layout = Layout::c_contiguous([2, 3]).expect("2x3 layout");
    let two = Layout::c_contiguous([2, 2]).expect("2x2 layout");
    assert!(
        ops.matmul_into(
            device,
            StridedView::new(&lhs, &lhs_layout),
            StridedView::new(&rhs, &two),
            StridedView::new(&out, &two),
        )
        .is_err(),
        "{name}: mismatched shared dimension must be rejected"
    );
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got, [9.0; 4],
        "{name}: rejected matmul must not mutate the output"
    );
}

/// Two independent 2x2 products in one batched dispatch.
fn batched_matmul_computes_per_batch_products<D, P>(device: &D, ops: &P)
where
    D: ComputeDevice,
    P: DenseProductOps<D, f32>,
{
    let name = device.backend_name();
    // Batch 0: identity times B; batch 1: [[1,2],[3,4]] times B.
    let lhs = device
        .upload(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0])
        .expect("lhs upload");
    let rhs = device
        .upload(&[5.0f32, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0])
        .expect("rhs upload");
    let out = device.alloc_zeroed::<f32>(8).expect("output alloc");
    let shape = Layout::c_contiguous([2, 2, 2]).expect("batched layout");
    ops.batched_matmul_into(
        device,
        StridedView::new(&lhs, &shape),
        StridedView::new(&rhs, &shape),
        StridedView::new(&out, &shape),
    )
    .expect("batched matmul dispatch");
    let mut got = [0.0f32; 8];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [5.0, 6.0, 7.0, 8.0, 19.0, 22.0, 43.0, 50.0],
        "{name}: per-batch exact products"
    );
}

/// `[[1,2],[3,4]] ⊗ [[0,1],[1,0]]` is the exact 4x4 block matrix.
fn kron_computes_the_exact_tensor_product<D, P>(device: &D, ops: &P)
where
    D: ComputeDevice,
    P: DenseProductOps<D, f32>,
{
    let name = device.backend_name();
    let lhs = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).expect("lhs upload");
    let rhs = device.upload(&[0.0f32, 1.0, 1.0, 0.0]).expect("rhs upload");
    let out = device.alloc_zeroed::<f32>(16).expect("output alloc");
    let two = Layout::c_contiguous([2, 2]).expect("2x2 layout");
    let four = Layout::c_contiguous([4, 4]).expect("4x4 layout");
    ops.kron_into(
        device,
        StridedView::new(&lhs, &two),
        StridedView::new(&rhs, &two),
        StridedView::new(&out, &four),
    )
    .expect("kron dispatch");
    let mut got = [0.0f32; 16];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [
            0.0, 1.0, 0.0, 2.0, //
            1.0, 0.0, 2.0, 0.0, //
            0.0, 3.0, 0.0, 4.0, //
            3.0, 0.0, 4.0, 0.0, //
        ],
        "{name}: exact Kronecker product"
    );
}
