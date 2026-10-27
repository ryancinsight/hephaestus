#![deny(missing_docs)]
//! Host reference device for the Hephaestus seams (ADR 0046).
//!
//! [`HostDevice`] implements [`ComputeDevice`](hephaestus_core::ComputeDevice)
//! over plain host memory so CPU reference implementors — leto adapters first —
//! can join the same role traits the GPU backends implement, and the
//! conformance suite can instantiate a CPU pair for every clause. This crate
//! is the **reference substrate**: correctness and conformance first, never a
//! performance path — consumers wanting fast CPU execution use leto directly.

/// Leto-backed rank-2 adaptive pooling (average / maximum) seam implementor.
pub mod adaptive_pooling;
/// Direct-loop argmax/argmin seam implementor (the reference for tie-break
/// semantics).
pub mod arg_reduce;
/// Leto-backed scaled dot-product attention seam implementor.
pub mod attention;
/// Leto-backed concatenation seam implementor.
pub mod cat;
mod combine;
/// Leto-backed regular and transposed convolution seam implementor.
pub mod convolution;
/// Leto-backed mean cross-entropy seam implementor.
pub mod cross_entropy;
/// Direct-loop batched cross-product seam implementor.
pub mod cross_product;
/// Direct-loop CTC step seam implementor.
pub mod ctc;
/// Leto as a decomposition-seam implementor.
pub mod decomposition;
/// Leto-backed dense product, composition, and matrix-function implementors.
pub mod dense_product;
/// Leto-backed dense-vector seam implementor.
pub mod dense_vector;
mod device;
/// Direct-loop inner-product seam implementor.
pub mod dot;
/// Host implementor of the unary/binary elementwise seams (ADR 0061).
pub mod elementwise;
/// Direct-loop embedding-table gather seam implementor.
pub mod embedding;
/// Leto-backed fixed-scheme three-dimensional sweep implementor.
pub mod fixed_fd;
/// Direct-loop n-D gather seam implementor.
pub mod gather;
/// Direct-loop n-D scatter-assignment seam implementor.
pub mod index_put;
/// Direct-loop n-D index-selection seam implementor.
pub mod index_select;
/// Leto-backed rank-2 axis resampling (nearest / linear) seam implementor.
pub mod interpolation;
/// Direct-loop n-D masked-fill seam implementor.
pub mod masked_fill;
/// Direct-loop nonzero-coordinate extraction implementor.
pub mod nonzero;
mod operands;
/// Leto-backed padding seam implementor.
pub mod pad;
/// Host implementor of the runtime-parameter unary elementwise seam (ADR
/// 0061).
pub mod parameterized;
/// Leto-backed pooling seam implementor.
pub mod pooling;
/// Leto-backed seeded random initialization seam implementor.
pub mod random;
/// Host implementor of the whole-operand and rank-2 axis reduction seams
/// (ADR 0061).
pub mod reduction;
/// Direct-loop rank-2 axis roll (circular shift) seam implementor.
pub mod roll;
/// Host implementor of the rank-2 axis scan seam (ADR 0061).
pub mod scan;
/// Direct-loop n-D scatter-add seam implementor.
pub mod scatter;
/// Leto-backed unfold/fold seam implementor.
pub mod sliding_window;
/// Leto-backed stable-sort seam implementor.
pub mod sort;
/// Leto-backed sparse-operator and batch-submit seam implementor.
pub mod sparse;
/// Leto-backed splitting seam implementor.
pub mod split;
/// Leto-backed stacking seam implementor.
pub mod stack;
/// Leto-backed staggered gradient/divergence implementor.
pub mod staggered;
/// Host implementor of the stateful parameter-update seam (ADR 0061).
pub mod stateful_update;
/// Leto-backed two-dimensional Laplacian stencil implementor.
pub mod stencil;
/// Direct-loop n-D tiling seam implementor.
pub mod tile;
/// Direct-insertion top-k selection seam implementor.
pub mod topk;
/// Direct-loop rank-2 triangular masking (tril / triu) seam implementor.
pub mod triangular;
/// Volume ray line integrals over leto interpolation.
pub mod volume;
/// Direct-loop n-D conditional-selection seam implementor.
pub mod where_cond;

pub use adaptive_pooling::HostAdaptivePoolingOps;
pub use arg_reduce::HostArgReduceOps;
pub use attention::{HostAttentionBackward, HostAttentionForward, HostAttentionOps};
pub use cat::HostCatOps;
pub use convolution::{
    HostConvolutionBackward, HostConvolutionForward, HostConvolutionOps,
    HostConvolutionTransposedBackward, HostConvolutionTransposedForward,
};
pub use cross_entropy::{HostCrossEntropyBackward, HostCrossEntropyForward, HostCrossEntropyOps};
pub use cross_product::HostCrossProductOps;
pub use ctc::HostCtcOps;
pub use decomposition::HostDecompositionOps;
pub use dense_product::HostDenseProductOps;
pub use dense_vector::{HostDenseVectorOps, HostPreparedDot, HostPreparedNorm};
pub use device::{HostBuffer, HostDevice};
pub use dot::HostDotOps;
pub use elementwise::{
    HostElementwiseOps, HostPreparedBinary, HostPreparedScalar, HostPreparedUnary,
};
pub use embedding::HostEmbeddingOps;
pub use fixed_fd::HostFixedFdOps;
pub use gather::HostGatherOps;
pub use index_put::HostIndexPutOps;
pub use index_select::HostIndexSelectOps;
pub use interpolation::HostInterpolationOps;
pub use masked_fill::HostMaskedFillOps;
pub use nonzero::HostNonzeroOps;
pub use pad::HostPadOps;
pub use parameterized::HostParameterizedUnaryOps;
pub use pooling::{HostPoolingBackward, HostPoolingForward, HostPoolingOps};
pub use random::HostRandomOps;
pub use reduction::{
    HostAxisReductionOps, HostFullReductionOps, HostPreparedAxisReduction,
    HostPreparedFullReduction,
};
pub use roll::HostRollOps;
pub use scan::{HostPreparedScan, HostScanOps};
pub use scatter::HostScatterOps;
pub use sliding_window::{HostSlidingWindowFold, HostSlidingWindowOps, HostSlidingWindowUnfold};
pub use sort::HostSortOps;
pub use sparse::{HostPreparedApply, HostSparseOps};
pub use split::HostSplitOps;
pub use stack::HostStackOps;
pub use staggered::HostStaggeredOps;
pub use stateful_update::HostStatefulUpdateOps;
pub use stencil::HostStencilOps;
pub use tile::HostTileOps;
pub use topk::HostTopKOps;
pub use triangular::HostTriangularOps;
pub use volume::HostRayIntegralOps;
pub use where_cond::HostWhereOps;

pub(crate) use device::{element_count, map_leto_error};
