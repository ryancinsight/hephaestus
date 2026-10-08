#![deny(missing_docs)]
//! # hephaestus-conformance
//!
//! One set of contract clauses that every accelerator backend is held to.
//!
//! Before this crate, each backend carried a hand-written `tests/contract.rs`
//! and the four had diverged: of the 112 entry points declared by all four
//! backends, only 46 were exercised by all four, and six were exercised by none
//! (Atlas conformance triage, 2026-07-28). The contract of a substitution seam
//! was in practice defined by whichever backend's author wrote the most tests.
//!
//! The clauses here are generic over
//! [`ComputeDevice`](hephaestus_core::ComputeDevice) and the operation seam, so
//! a backend runs them by instantiating rather than by re-authoring, and a
//! clause added once is executed by every backend from then on.
//!
//! ## Shape
//!
//! Every clause is a free function taking the device and the seam value, and
//! panicking with a located message on violation — the shape a test harness
//! expects, without this crate depending on one. Clauses are grouped by the seam
//! they exercise, one module per seam.
//!
//! ## Oracles
//!
//! Clauses assert exact equality wherever the arithmetic admits it, and derive
//! any tolerance they do need from the operation rather than from observation.
//! The reduction clauses use integer-valued `f32` operands whose products stay
//! below `2^24`, where `f32` represents every intermediate exactly and addition
//! and multiplication are associative over the values involved — so reduction
//! order cannot change the result and no tolerance is applicable. A clause that
//! needed an epsilon would state its derivation at the assertion site.

/// Contract clauses for the
/// [`AdaptivePoolingOps`](hephaestus_core::AdaptivePoolingOps) seam.
pub mod adaptive_pooling;
/// Contract clauses for the [`ArgReduceOps`](hephaestus_core::ArgReduceOps) seam.
pub mod arg_reduce;
/// The aggregate entry point: every seam named as one bound.
pub mod assert_backend;
/// Contract clauses for the [`AttentionOps`](hephaestus_core::AttentionOps) seam.
pub mod attention;
/// Contract clauses for the [`AxisReductionOps`](hephaestus_core::AxisReductionOps) seam.
pub mod axis_reduction;
/// Contract clauses for the backend-abstracted blocked Cholesky loop
/// ([`BlockedCholeskyBackend`](hephaestus_core::BlockedCholeskyBackend)).
pub mod blocked_cholesky;
/// Contract clause for the backend-abstracted blocked LU loop
/// ([`BlockedDecompositionBackend`](hephaestus_core::BlockedDecompositionBackend)).
pub mod blocked_lu;
/// Contract clauses for the backend-abstracted blocked QR loop
/// ([`BlockedQrBackend`](hephaestus_core::BlockedQrBackend)).
pub mod blocked_qr;
/// Contract clauses for the [`ConvolutionOps`](hephaestus_core::ConvolutionOps) seam.
pub mod convolution;
/// Contract clauses for the [`CrossEntropyOps`](hephaestus_core::CrossEntropyOps) seam.
pub mod cross_entropy;
/// Contract clauses for the [`CrossProductOps`](hephaestus_core::CrossProductOps) seam.
pub mod cross_product;
/// Contract clauses for the [`CtcOps`](hephaestus_core::CtcOps) seam.
pub mod ctc;
/// Contract clauses for the
/// [`DecompositionOps`](hephaestus_core::DecompositionOps) seam.
pub mod decomposition;
/// Contract clauses for the
/// [`DenseCompositionOps`](hephaestus_core::DenseCompositionOps) seam.
pub mod dense_composition;
/// Contract clauses for the
/// [`DenseMatrixFunctionOps`](hephaestus_core::DenseMatrixFunctionOps) seam.
pub mod dense_matrix_function;
/// Contract clauses for the
/// [`DenseVectorOps`](hephaestus_core::DenseVectorOps) seam.
/// Dense matmul/batched-matmul/Kronecker product clauses.
pub mod dense_product;
pub mod dense_vector;
/// Untyped unary/binary elementwise arithmetic clauses.
pub mod elementwise;
/// Contract clauses for the
/// [`EmbeddingOps`](hephaestus_core::EmbeddingOps) seam.
pub mod embedding;
/// Contract clauses for the
/// [`FixedFd3DOps`](hephaestus_core::FixedFd3DOps) seam.
pub mod fixed_fd;
/// Contract clauses for the
/// [`FullReductionOps`](hephaestus_core::FullReductionOps) seam.
pub mod full_reduction;
/// Contract clauses for the
/// [`InterpolationOps`](hephaestus_core::InterpolationOps) seam.
pub mod interpolation;
/// Contract clauses for the device-neutral padding seam.
pub mod pad;
/// Contract clauses for runtime-parameter unary dispatch.
pub mod parameterized_unary;
/// Seeded random initialization clauses.
pub mod random_init;
/// Contract clauses for the
/// [`RayIntegralOps`](hephaestus_core::RayIntegralOps) seam.
pub mod ray_integral;
/// Contract clauses for the [`ScanOps`](hephaestus_core::ScanOps) seam.
pub mod scan;
/// Contract clauses for the
/// [`SparseOperatorOps`](hephaestus_core::SparseOperatorOps) seam.
pub mod sparse;
/// 2D stencil clauses with an analytical quadratic oracle.
pub mod staggered;
/// Contract clauses for provider-owned stateful parameter updates.
pub mod stateful_update;
pub mod stencil;
/// Contract clauses for the [`TopKOps`](hephaestus_core::TopKOps) seam.
pub mod topk;
/// Device transfer and buffer-initialization clauses.
pub mod transfer;
/// Contract clauses for the
/// [`TriangularOps`](hephaestus_core::TriangularOps) seam.
pub mod triangular;
/// Contract clauses for the typed paths of the
/// [`ElementwiseOps`](hephaestus_core::ElementwiseOps) seam.
pub mod typed_elementwise;

pub use adaptive_pooling::assert_adaptive_pooling_contract;
pub use arg_reduce::{assert_arg_reduce_contract, assert_arg_reduce_transposed_view_contract};
pub use assert_backend::{BackendUnderTest, assert_backend_contract};
pub use attention::assert_attention_contract;
pub use axis_reduction::assert_axis_reduction_contract;
pub use blocked_cholesky::assert_blocked_cholesky_contract;
pub use blocked_lu::assert_blocked_lu_contract;
pub use blocked_qr::assert_blocked_qr_contract;
pub use convolution::{assert_convolution_contract, assert_convolution_f64_contract};
pub use cross_entropy::{
    assert_cross_entropy_backward_contract, assert_cross_entropy_backward_contract_f64,
    assert_cross_entropy_contract, assert_cross_entropy_contract_f64,
};
pub use cross_product::assert_cross_product_contract;
pub use ctc::assert_ctc_contract;
pub use decomposition::assert_decomposition_contract;
pub use dense_composition::assert_dense_composition_contract;
pub use dense_matrix_function::assert_dense_matrix_function_contract;
pub use dense_product::assert_dense_product_contract;
pub use dense_vector::assert_dense_vector_contract;
pub use elementwise::{assert_bessel_contract, assert_elementwise_contract, assert_sinc_contract};
pub use embedding::{
    assert_embedding_gather_contract, assert_embedding_gather_rejects_out_of_range_index,
};
pub use fixed_fd::assert_fixed_fd_3d_contract;
pub use full_reduction::assert_full_reduction_contract;
pub use interpolation::assert_interpolation_contract;
pub use pad::assert_pad_contract;
pub use parameterized_unary::assert_parameterized_unary_contract;
pub use random_init::assert_random_init_contract;
pub use ray_integral::assert_ray_integral_contract;
pub use scan::{assert_scan_contract, assert_scan_leto_contract};
pub use sparse::{assert_batch_submit_contract, assert_sparse_operator_contract};
pub use staggered::assert_staggered_3d_contract;
pub use stateful_update::{assert_stateful_update_contract, assert_stateful_update_contract_f64};
pub use stencil::assert_stencil_contract;
pub use topk::assert_topk_contract;
pub use transfer::assert_transfer_contract;
pub use triangular::assert_triangular_contract;
pub use typed_elementwise::assert_typed_elementwise_contract;
