//! Pipeline-cache identity for generated spatial-window kernels.

use core::any::TypeId;

/// A generated spatial-window kernel specialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WindowOperation {
    /// Forward maximum pooling.
    PoolingForwardMaximum,
    /// Forward average pooling.
    PoolingForwardAverage,
    /// Maximum-pooling input gradient accumulation.
    PoolingBackwardMaximum,
    /// Average-pooling input gradient accumulation.
    PoolingBackwardAverage,
    /// Extract spatial windows into column storage.
    Unfold,
    /// Accumulate column storage into a spatial tensor.
    Fold,
}

/// Pipeline-cache identity for one generated spatial-window kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WindowKey {
    /// Operation and pooling mode represented by the source.
    pub operation: WindowOperation,
    /// Host scalar represented by the source.
    pub scalar: TypeId,
    /// Number of spatial axes represented by the metadata.
    pub spatial_rank: usize,
}
