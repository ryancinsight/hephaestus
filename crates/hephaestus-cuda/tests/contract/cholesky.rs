use super::*;

/// The blocked-Cholesky differential is one shared clause (SUBSTRATE-003):
/// the fixtures and assertions live in `hephaestus-conformance`, generic over
/// the `BlockedCholeskyBackend` seam the shared `blocked_cholesky` loop drives.
#[cfg(feature = "decomposition")]
#[test]
fn blocked_cholesky_contract() {
    let Some(dev) = device("blocked_cholesky_contract") else {
        return;
    };
    const BLOCK_SIZE: usize = 64;
    hephaestus_conformance::assert_blocked_cholesky_contract(&dev, BLOCK_SIZE);
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_cholesky_solve_known_system_accurate() {
    let Some(dev) = device("blocked_cholesky_solve_known_system_accurate") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, cholesky_decompose_blocked};
    use leto::Layout;

    // A = [[4, 2], [2, 3]], b = [8, 7]  =>  x = [1.25, 1.5]
    let matrix_host = vec![4.0f32, 2.0, 2.0, 3.0];
    let rhs_host = vec![8.0f32, 7.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let rhs = dev.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();

    let gpu_chol = cholesky_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_chol.solve(&dev, &rhs).unwrap();
    let mut got = vec![0.0f32; 2];
    dev.download(&solution, &mut got).unwrap();
    assert!(
        (got[0] - 1.25f32).abs() <= 1e-5,
        "x[0] = {} expected 1.25",
        got[0]
    );
    assert!(
        (got[1] - 1.5f32).abs() <= 1e-5,
        "x[1] = {} expected 1.5",
        got[1]
    );

    // Verify residual A*x ≈ b.
    let ax0 = 4.0 * got[0] + 2.0 * got[1];
    let ax1 = 2.0 * got[0] + 3.0 * got[1];
    assert!((ax0 - 8.0).abs() <= 1e-4, "residual[0] = {ax0}");
    assert!((ax1 - 7.0).abs() <= 1e-4, "residual[1] = {ax1}");
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_cholesky_rejects_non_dense_operands() {
    let Some(dev) = device("blocked_cholesky_rejects_non_dense_operands") else {
        return;
    };
    use hephaestus_cuda::cholesky_decompose_blocked;
    assert_blocked_rejects_non_dense(&dev, cholesky_decompose_blocked, "cholesky");
}
