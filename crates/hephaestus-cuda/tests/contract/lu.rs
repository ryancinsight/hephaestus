use super::*;

/// The blocked-LU differential is one shared clause (SUBSTRATE-003): the
/// fixtures and assertions live in `hephaestus-conformance`, generic over the
/// `BlockedDecompositionBackend` seam the shared `blocked_lu` loop drives.
/// This instantiation supplies the device and the panel width the blocked
/// entry points factor with.
#[cfg(feature = "decomposition")]
#[test]
fn blocked_lu_contract() {
    let Some(dev) = device("blocked_lu_contract") else {
        return;
    };
    const BLOCK_SIZE: usize = 64;
    hephaestus_conformance::assert_blocked_lu_contract(&dev, BLOCK_SIZE);
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_pivoted_decompositions_match_ordinary_contracts() {
    let Some(dev) = device("blocked_pivoted_decompositions_match_ordinary_contracts") else {
        return;
    };
    use hephaestus_cuda::{
        StridedOperand, col_piv_qr, col_piv_qr_blocked, full_piv_lu, full_piv_lu_blocked,
    };
    use leto::Layout;

    let square_host = [2.0f32, 5.0, -2.0, 1.0, 2.0, 3.0, -2.0, 4.0, 3.0];
    let square = dev.upload(&square_host).unwrap();
    let square_layout = Layout::c_contiguous([3, 3]).unwrap();
    let square_operand = StridedOperand {
        buffer: &square,
        layout: &square_layout,
    };
    let ordinary_lu = full_piv_lu(&dev, square_operand).unwrap();
    let blocked_lu = full_piv_lu_blocked(&dev, square_operand).unwrap();
    assert_eq!(blocked_lu.rank(), ordinary_lu.rank());
    assert_eq!(blocked_lu.row_permutation(), ordinary_lu.row_permutation());
    assert_eq!(blocked_lu.col_permutation(), ordinary_lu.col_permutation());
    assert_close(blocked_lu.det(), ordinary_lu.det(), 1.0e-5);
    let mut ordinary_factors = vec![0.0f32; 9];
    let mut blocked_factors = vec![0.0f32; 9];
    dev.download(ordinary_lu.lu_buffer(), &mut ordinary_factors)
        .unwrap();
    dev.download(blocked_lu.lu_buffer(), &mut blocked_factors)
        .unwrap();
    assert_close_slice(&blocked_factors, &ordinary_factors, 1.0e-5, 0.0);

    let tall_host = [1.0f32, 0.0, 0.0, 2.0, 0.0, 0.0];
    let tall = dev.upload(&tall_host).unwrap();
    let tall_layout = Layout::c_contiguous([3, 2]).unwrap();
    let tall_operand = StridedOperand {
        buffer: &tall,
        layout: &tall_layout,
    };
    let ordinary_qr = col_piv_qr(&dev, tall_operand).unwrap();
    let blocked_qr = col_piv_qr_blocked(&dev, tall_operand).unwrap();
    assert_eq!(blocked_qr.rank(), ordinary_qr.rank());
    assert_eq!(blocked_qr.permutation(), ordinary_qr.permutation());
    let mut ordinary_q = vec![0.0f32; 9];
    let mut blocked_q = vec![0.0f32; 9];
    let mut ordinary_r = vec![0.0f32; 6];
    let mut blocked_r = vec![0.0f32; 6];
    dev.download(ordinary_qr.q(), &mut ordinary_q).unwrap();
    dev.download(blocked_qr.q(), &mut blocked_q).unwrap();
    dev.download(ordinary_qr.r(), &mut ordinary_r).unwrap();
    dev.download(blocked_qr.r(), &mut blocked_r).unwrap();
    assert_close_slice(&blocked_q, &ordinary_q, 1.0e-5, 0.0);
    assert_close_slice(&blocked_r, &ordinary_r, 1.0e-5, 0.0);
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_lu_rejects_non_dense_operands() {
    let Some(dev) = device("blocked_lu_rejects_non_dense_operands") else {
        return;
    };
    use hephaestus_cuda::lu_decompose_blocked;
    assert_blocked_rejects_non_dense(&dev, lu_decompose_blocked, "LU");
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_pivoted_decompositions_reject_non_dense_operands() {
    let Some(dev) = device("blocked_pivoted_decompositions_reject_non_dense_operands") else {
        return;
    };
    assert_blocked_rejects_non_dense(
        &dev,
        hephaestus_cuda::full_piv_lu_blocked,
        "complete-pivoted LU",
    );
    assert_blocked_rejects_non_dense(
        &dev,
        hephaestus_cuda::col_piv_qr_blocked,
        "column-pivoted QR",
    );
}

#[cfg(feature = "decomposition")]
#[test]
fn empty_decompositions_preserve_shapes_and_identities() {
    use hephaestus_cuda::{
        bidiagonalize, col_piv_qr, full_piv_lu, hessenberg, qr_decompose, qr_decompose_blocked,
    };

    let Some(dev) = device("empty_decompositions_preserve_shapes_and_identities") else {
        return;
    };
    let empty = dev.alloc_zeroed::<f32>(0).unwrap();
    let tall_layout = Layout::c_contiguous([3, 0]).unwrap();
    let tall = StridedOperand {
        buffer: &empty,
        layout: &tall_layout,
    };

    let bidiagonal = bidiagonalize(&dev, tall).unwrap();
    assert_eq!(bidiagonal.shape(), (3, 0));
    assert_eq!(bidiagonal.u_buffer().len(), 9);
    assert_eq!(bidiagonal.b_buffer().len(), 0);
    assert_eq!(bidiagonal.v_buffer().len(), 0);
    let expected_identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut identity = vec![0.0; 9];
    dev.download(bidiagonal.u_buffer(), &mut identity).unwrap();
    assert_eq!(identity, expected_identity);

    let pivoted = col_piv_qr(&dev, tall).unwrap();
    assert_eq!(pivoted.rank(), 0);
    assert_eq!(pivoted.permutation(), &[] as &[usize]);
    assert_eq!(pivoted.q().len(), 9);
    assert_eq!(pivoted.r().len(), 0);
    identity.fill(0.0);
    dev.download(pivoted.q(), &mut identity).unwrap();
    assert_eq!(identity, expected_identity);

    for qr in [
        qr_decompose(&dev, tall).unwrap(),
        qr_decompose_blocked(&dev, tall).unwrap(),
    ] {
        assert_eq!(qr.shape(), (3, 0));
        assert_eq!(qr.r_buffer().len(), 0);
        assert_eq!(qr.inner().shape(), (3, 0));
        assert_eq!(
            leto::Storage::as_slice(qr.inner().q().storage()),
            expected_identity
        );
        assert_eq!(qr.inner().r().shape(), [3, 0]);
    }

    let square_layout = Layout::c_contiguous([0, 0]).unwrap();
    let square = StridedOperand {
        buffer: &empty,
        layout: &square_layout,
    };
    let lu = full_piv_lu(&dev, square).unwrap();
    assert_eq!(lu.n(), 0);
    assert_eq!(lu.rank(), 0);
    assert_eq!(lu.det(), 1.0);
    assert_eq!(lu.row_permutation(), &[] as &[usize]);
    assert_eq!(lu.col_permutation(), &[] as &[usize]);
    assert_eq!(lu.lu_buffer().len(), 0);

    let h = hessenberg(&dev, square).unwrap();
    assert_eq!(h.n(), 0);
    assert_eq!(h.q_buffer().len(), 0);
    assert_eq!(h.h_buffer().len(), 0);
}
