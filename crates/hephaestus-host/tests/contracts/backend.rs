//! The host device held to the whole backend contract in one instantiation.
//!
//! The per-seam modules beside this one each run their own clause, which
//! leaves coverage opt-in: a seam nobody wires reads as green. Naming every
//! seam in one [`BackendUnderTest`] closes that hole for the host, since a
//! missing implementation is a compile error at this call site rather than a
//! clause that silently stops running.

use hephaestus_conformance::{BackendUnderTest, assert_backend_contract};
use hephaestus_host::{
    HostAttentionOps, HostAxisReductionOps, HostConvolutionOps, HostCrossEntropyOps,
    HostDecompositionOps, HostDenseProductOps, HostDenseVectorOps, HostDevice, HostElementwiseOps,
    HostFullReductionOps, HostParameterizedUnaryOps, HostRandomOps, HostRayIntegralOps,
    HostScanOps, HostSparseOps, HostStaggeredOps, HostStatefulUpdateOps, HostStencilOps,
};

#[test]
fn host_satisfies_the_backend_contract() {
    let device = HostDevice::new();
    assert_backend_contract(&BackendUnderTest {
        device: &device,
        attention: &HostAttentionOps,
        axis_reduction: &HostAxisReductionOps,
        convolution: &HostConvolutionOps,
        cross_entropy: &HostCrossEntropyOps,
        decomposition: &HostDecompositionOps,
        dense_product: &HostDenseProductOps,
        dense_vector: &HostDenseVectorOps,
        elementwise: &HostElementwiseOps,
        full_reduction: &HostFullReductionOps,
        parameterized_unary: &HostParameterizedUnaryOps,
        random_init: &HostRandomOps,
        ray_integral: &HostRayIntegralOps,
        scan: &HostScanOps,
        sparse_operator: &HostSparseOps,
        batch_submit: &HostSparseOps,
        staggered_3d: &HostStaggeredOps,
        stateful_update: &HostStatefulUpdateOps,
        stencil: &HostStencilOps,
        typed_elementwise: &HostElementwiseOps,
    });
}
