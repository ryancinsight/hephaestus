use hephaestus_core::WindowOperation;
use std::any::TypeId;

/// Pipeline-cache key for a runtime-compiled ROCm kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
        direction: hephaestus_core::ScanDirection,
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
    Kron {
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
    #[cfg(feature = "decomposition")]
    Cholesky(CholeskyStage),
    #[cfg(feature = "decomposition")]
    Lu(LuStage),
    #[cfg(feature = "decomposition")]
    Qr(QrStage),
    #[cfg(feature = "decomposition")]
    FullPivLu(FullPivLuStage),
    #[cfg(feature = "decomposition")]
    ColPivQr(ColPivQrStage),
    MultiStorage(u64),
    Stream(u64),
    #[cfg(all(feature = "rocm", target_os = "linux"))]
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
    },
    GroupedStream(u64),
}

#[cfg(feature = "decomposition")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CholeskyStage {
    Validate,
    Diagonal,
    Column,
    ClearUpper,
}

#[cfg(feature = "decomposition")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum LuStage {
    Validate,
    Step,
}

#[cfg(feature = "decomposition")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum QrStage {
    Validate,
    Step,
}

#[cfg(feature = "decomposition")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FullPivLuStage {
    Validate,
    Step,
}

#[cfg(feature = "decomposition")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ColPivQrStage {
    Validate,
    Step,
}
