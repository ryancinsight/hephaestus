use hephaestus_core::ScanDirection;
use std::any::TypeId;

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
