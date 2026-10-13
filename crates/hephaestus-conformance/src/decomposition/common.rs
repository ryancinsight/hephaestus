//! Shared decomposition-clause support: the derived bound and the
//! multi-family identity clause.
//!
//! Single-family helpers live with their families; only genuinely shared
//! items are here.

use hephaestus_core::{
    CholeskyHandle, ComputeDevice, DecompositionOps, LuHandle, QrHandle, StridedView,
};
use leto::Layout;

/// Derived clause bound; see the module-level derivation.
pub(crate) const DECOMP_BOUND: f32 = 1e-3;

/// Factorizing the identity is exact: every arithmetic step multiplies by
/// 0 or 1 or takes `sqrt(1)`, so no rounding occurs and the factors admit
/// exact equality (LU packed factors and pivots, Cholesky lower). QR
/// asserts entry magnitudes within the derived bound only: the Householder
/// sign convention belongs to the backend and is not contract.
pub(crate) fn identity_factorizations_are_exact<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let identity = [1.0f32, 0.0, 0.0, 1.0];
    let layout = Layout::c_contiguous([2, 2]).expect("layout");

    let a = device.upload(&identity).expect("upload");
    let lu = ops
        .lu(device, StridedView::new(&a, &layout))
        .expect("identity LU");
    let mut factors = [0.0f32; 4];
    device
        .download(lu.factors(), &mut factors)
        .expect("factor download");
    assert_eq!(factors, identity, "{name}: identity LU factors");
    assert_eq!(lu.pivots(), &[0, 1], "{name}: identity LU pivots");
    assert!(
        (lu.det() - 1.0).abs() == 0.0,
        "{name}: identity LU det {}",
        lu.det()
    );

    let a = device.upload(&identity).expect("upload");
    let chol = ops
        .cholesky(device, StridedView::new(&a, &layout))
        .expect("identity Cholesky");
    let mut lower = [0.0f32; 4];
    device
        .download(chol.lower(), &mut lower)
        .expect("factor download");
    assert_eq!(lower, identity, "{name}: identity Cholesky lower");

    let a = device.upload(&identity).expect("upload");
    let qr = ops
        .qr(device, StridedView::new(&a, &layout))
        .expect("identity QR");
    use hephaestus_core::DeviceBuffer;
    let mut r = vec![0.0f32; qr.r_buffer().len()];
    device.download(qr.r_buffer(), &mut r).expect("R download");
    for (index, value) in r.iter().take(4).enumerate() {
        let expected = identity[index];
        assert!(
            (value.abs() - expected).abs() < DECOMP_BOUND,
            "{name}: identity QR R entry {index} = {value}, expected magnitude {expected}"
        );
    }
}
