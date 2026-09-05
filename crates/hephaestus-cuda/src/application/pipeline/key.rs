use hephaestus_core::{FusedReduction, ScanDirection, WindowOperation};
use std::any::TypeId;
use std::sync::Arc;

/// Pipeline-cache identity for one compiled CUDA kernel specialization.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PipelineKey {
    Binary {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    Scalar {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    Unary {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    Reduction {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    MapReduction {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    AxisReduction {
        op: TypeId,
        scalar: TypeId,
        axis: usize,
        width: u32,
    },
    MeanAxis {
        scalar: TypeId,
        axis: usize,
        width: u32,
    },
    AxisScan {
        marker: TypeId,
        scalar: TypeId,
        direction: ScanDirection,
        axis: usize,
        width: u32,
    },
    /// A generated spatial-window kernel: pooling forward and backward,
    /// unfold and fold. The accelerator layer plans one kernel per
    /// (operation, host scalar, spatial rank), so those three identify it.
    Window {
        operation: WindowOperation,
        scalar: TypeId,
        spatial_rank: usize,
    },
    Kron {
        marker: TypeId,
        scalar: TypeId,
    },
    Matmul {
        marker: TypeId,
        scalar: TypeId,
    },
    MatrixIdentity {
        scalar: TypeId,
    },
    BatchedMatmul {
        marker: TypeId,
        scalar: TypeId,
    },
    MatrixRank {
        marker: TypeId,
        scalar: TypeId,
    },
    Spmm {
        marker: TypeId,
        scalar: TypeId,
    },
    Spmv {
        marker: TypeId,
        scalar: TypeId,
    },
    StridedBinary {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    StridedUnary {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    ParameterizedStridedUnary {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    StatefulUpdate {
        rule: TypeId,
        scalar: TypeId,
        width: u32,
    },
    StridedScalar {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    #[cfg(all(feature = "cuda", feature = "decomposition"))]
    CholeskySyrk,
    #[cfg(all(feature = "cuda", feature = "decomposition"))]
    LuGemm,
    #[cfg(all(feature = "cuda", feature = "decomposition"))]
    QrHouseholder,
    #[cfg(all(feature = "cuda", feature = "decomposition"))]
    QrAccumulateQ,
    #[cfg(all(feature = "cuda", feature = "decomposition"))]
    SplitPackedLu,
    Stream(u64),
    GroupedStream(u64),
    MultiStorage(u64),
    Convolution {
        entry: &'static str,
        scalar: TypeId,
        spatial_rank: usize,
        bias: bool,
    },
    Attention {
        entry: &'static str,
        scalar: TypeId,
    },
    CrossEntropy {
        entry: &'static str,
        scalar: TypeId,
    },
}

/// Collision-safe identity for a runtime-generated fusion kernel.
///
/// The complete expression is retained in the key. A truncated digest would
/// let two distinct expressions reuse the wrong compiled kernel.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FusionPipelineKey {
    pub(crate) family: TypeId,
    pub(crate) scalar: TypeId,
    pub(crate) rank: u32,
    pub(crate) input_count: u32,
    pub(crate) reduction: Option<FusedReduction>,
    pub(crate) expression: Arc<str>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_key_compares_complete_expression_source() {
        let base = FusionPipelineKey {
            family: TypeId::of::<()>(),
            scalar: TypeId::of::<f32>(),
            rank: 2,
            input_count: 2,
            reduction: None,
            expression: Arc::from("input_0 + input_1"),
        };
        let changed = FusionPipelineKey {
            expression: Arc::from("input_0 - input_1"),
            ..base.clone()
        };
        assert_ne!(base, changed);
        assert_eq!(base, base.clone());
    }
}
