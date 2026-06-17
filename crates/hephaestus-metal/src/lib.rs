#![deny(missing_docs)]
//! # hephaestus-metal
//!
//! The Metal backend of the Atlas accelerator substrate (atlas ADR 0001).
//! Implements the `hephaestus-core` [`ComputeDevice`] seam by delegating to
//! `hephaestus-wgpu` configured to use the native Metal API.

/// Compute dispatch delegation to Metal.
pub mod application;
/// Metal device and buffer infrastructure.
pub mod infrastructure;

pub use application::elementwise::{
    binary_elementwise, binary_elementwise_into, scalar_elementwise, scalar_elementwise_into,
    unary_elementwise, unary_elementwise_into, AbsOp, AddOp, CosOp, DivOp, ExpOp, IdentityOp, LnOp,
    MulOp, NegOp, PowOp, RecipOp, SinOp, SqrtOp, SubOp,
};
pub use application::linalg::{
    batched_matmul, batched_matmul_into, det, dot, kron, kron_into, matexp, matmul, matmul_into,
    matpow, matrix_rank, matrix_rank_with_tolerance, norm_l1, norm_l2, norm_max, pinv, trace,
};
pub use application::reduction::{
    max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis, min_axis_into, reduce_axis,
    reduction, reduction_with_width, sum_axis, sum_axis_into, MaxOp, MinOp, SumOp,
};
pub use application::scan::{
    cumsum, cumsum_into, scan_axis, scan_axis_into, CumProdOp, CumSumOp, ScanDirection,
};
pub use application::strided::{
    binary_elementwise_strided, binary_elementwise_strided_into, scalar_elementwise_strided,
    scalar_elementwise_strided_into, unary_elementwise_strided, unary_elementwise_strided_into,
    StridedOperand, MAX_STRIDED_RANK,
};
pub use infrastructure::buffer::MetalBuffer;
pub use infrastructure::device::MetalDevice;

#[cfg(feature = "decomposition")]
pub use application::decomposition::{
    bidiagonalize, bunch_kaufman, cholesky_decompose, cholesky_decompose_blocked, col_piv_qr,
    eigenvalues, full_piv_lu, hessenberg, lu_decompose, lu_decompose_blocked, qr_decompose,
    qr_decompose_blocked, schur, singular_values, svd_decompose, svd_rank_revealing,
    symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi, udu_decompose,
    GpuBidiagonalDecomposition, GpuBunchKaufmanDecomposition, GpuCholesky,
    GpuColPivQrDecomposition, GpuFullPivLuDecomposition, GpuHessenbergDecomposition,
    GpuLuDecomposition, GpuQrDecomposition, GpuRealSchur, GpuSvdDecomposition,
    GpuSymmetricEigenDecomposition, GpuUduDecomposition,
};

pub use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};
