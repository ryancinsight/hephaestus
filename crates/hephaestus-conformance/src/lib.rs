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

/// Contract clauses for the [`AttentionOps`](hephaestus_core::AttentionOps) seam.
pub mod attention;
/// Contract clauses for the [`AxisReductionOps`](hephaestus_core::AxisReductionOps) seam.
pub mod axis_reduction;
/// Contract clauses for the [`ConvolutionOps`](hephaestus_core::ConvolutionOps) seam.
pub mod convolution;
/// Contract clauses for the [`CrossEntropyOps`](hephaestus_core::CrossEntropyOps) seam.
pub mod cross_entropy;
/// Contract clauses for the
/// [`DecompositionOps`](hephaestus_core::DecompositionOps) seam.
pub mod decomposition;
/// Contract clauses for the
/// [`DenseVectorOps`](hephaestus_core::DenseVectorOps) seam.
/// Dense matmul/batched-matmul/Kronecker product clauses.
pub mod dense_product;
pub mod dense_vector;
/// Untyped unary/binary elementwise arithmetic clauses.
pub mod elementwise;
/// Contract clauses for the
/// [`FullReductionOps`](hephaestus_core::FullReductionOps) seam.
pub mod full_reduction;
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
/// Device transfer and buffer-initialization clauses.
pub mod transfer;
/// Contract clauses for the typed paths of the
/// [`ElementwiseOps`](hephaestus_core::ElementwiseOps) seam.
pub mod typed_elementwise;

pub use attention::assert_attention_contract;
pub use axis_reduction::assert_axis_reduction_contract;
pub use convolution::{assert_convolution_contract, assert_convolution_f64_contract};
pub use cross_entropy::assert_cross_entropy_contract;
pub use decomposition::assert_decomposition_contract;
pub use dense_product::assert_dense_product_contract;
pub use dense_vector::assert_dense_vector_contract;
pub use elementwise::assert_elementwise_contract;
pub use full_reduction::assert_full_reduction_contract;
pub use parameterized_unary::assert_parameterized_unary_contract;
pub use random_init::assert_random_init_contract;
pub use ray_integral::assert_ray_integral_contract;
pub use scan::assert_scan_contract;
pub use sparse::{assert_batch_submit_contract, assert_sparse_operator_contract};
pub use staggered::assert_staggered_3d_contract;
pub use stateful_update::assert_stateful_update_contract;
pub use stencil::assert_stencil_contract;
pub use transfer::assert_transfer_contract;
pub use typed_elementwise::assert_typed_elementwise_contract;
