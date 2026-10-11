//! Device-neutral runtime-rank expression-fusion contracts.
//!
//! Static operation markers remain the right representation for a fixed
//! operation family. Consumer expression graphs are different: their source,
//! rank, and input count are runtime values. This module carries only that
//! variation across the provider boundary; a backend still monomorphizes the
//! dispatch contract over its device, scalar, dialect, and expression adapter.
//!
//! The C-dialect kernel source is owned here once and monomorphized per
//! backend dialect, so CUDA and HIP compile byte-identical kernels from one
//! generator instead of carrying a per-vendor fork.

mod c_source;
mod ops;
mod scalar;

pub use c_source::{
    FusionLayoutInfo, elementwise_source, fusion_expression_shape, reduction_source,
    validate_c_expression_source,
};
pub use ops::{
    FUSION_WORKGROUP_WIDTH, FusedElementwiseOps, FusedExpression, FusedReduction,
    FusedReductionOps, FusionPipelineKey, MAX_FUSION_RANK,
};
pub use scalar::FusionScalar;
