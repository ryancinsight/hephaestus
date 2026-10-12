//! The host device held to the whole backend contract in one instantiation.
//!
//! The per-seam modules beside this one each run their own clause, which
//! leaves coverage opt-in: a seam nobody wires reads as green. Naming every
//! seam in one [`BackendUnderTest`] closes that hole for the host, since a
//! missing implementation is a compile error at this call site rather than a
//! clause that silently stops running.

use hephaestus_conformance::{BackendUnderTest, assert_backend_contract};
use hephaestus_host::{
    HostAdaptivePoolingOps, HostArgReduceOps, HostAttentionOps, HostAxisReductionOps, HostCatOps,
    HostConvolutionOps, HostCrossEntropyOps, HostCrossProductOps, HostCtcOps, HostDecompositionOps,
    HostDenseProductOps, HostDenseVectorOps, HostDevice, HostDotOps, HostElementwiseOps,
    HostEmbeddingOps, HostFixedFdOps, HostFullReductionOps, HostGatherOps, HostIndexPutOps,
    HostIndexSelectOps, HostInterpolationOps, HostMaskedFillOps, HostNonzeroOps, HostPadOps,
    HostParameterizedUnaryOps, HostRandomOps, HostRayIntegralOps, HostRollOps, HostScanOps,
    HostScatterOps, HostSortOps, HostSparseOps, HostSplitOps, HostStackOps, HostStaggeredOps,
    HostStatefulUpdateOps, HostStencilOps, HostTileOps, HostTopKOps, HostTriangularOps,
    HostWhereOps,
};

#[test]
fn host_satisfies_the_backend_contract() {
    let device = HostDevice::new();
    assert_backend_contract(&BackendUnderTest {
        device: &device,
        adaptive_pooling: &HostAdaptivePoolingOps,
        arg_reduce: &HostArgReduceOps,
        attention: &HostAttentionOps,
        axis_reduction: &HostAxisReductionOps,
        cat: &HostCatOps,
        convolution: &HostConvolutionOps,
        cross_entropy: &HostCrossEntropyOps,
        cross_product: &HostCrossProductOps,
        ctc: &HostCtcOps,
        decomposition: &HostDecompositionOps,
        dense_composition: &HostDenseProductOps,
        dense_matrix_function: &HostDenseProductOps,
        dense_product: &HostDenseProductOps,
        dense_vector: &HostDenseVectorOps,
        elementwise: &HostElementwiseOps,
        embedding: &HostEmbeddingOps,
        gather: &HostGatherOps,
        fixed_fd_3d: &HostFixedFdOps,
        full_reduction: &HostFullReductionOps,
        interpolation: &HostInterpolationOps,
        pad: &HostPadOps,
        parameterized_unary: &HostParameterizedUnaryOps,
        random_init: &HostRandomOps,
        ray_integral: &HostRayIntegralOps,
        scan: &HostScanOps,
        sparse_operator: &HostSparseOps,
        split: &HostSplitOps,
        stack: &HostStackOps,
        scatter: &HostScatterOps,
        index_select: &HostIndexSelectOps,
        index_put: &HostIndexPutOps,
        masked_fill: &HostMaskedFillOps,
        roll: &HostRollOps,
        tile: &HostTileOps,
        where_cond: &HostWhereOps,
        sort: &HostSortOps,
        nonzero: &HostNonzeroOps,
        dot: &HostDotOps,
        batch_submit: &HostSparseOps,
        staggered_3d: &HostStaggeredOps,
        stateful_update: &HostStatefulUpdateOps,
        stencil: &HostStencilOps,
        topk: &HostTopKOps,
        triangular: &HostTriangularOps,
        typed_elementwise: &HostElementwiseOps,
    });
}
