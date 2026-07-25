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
    AbsOp, AddOp, CosOp, DivOp, ExpOp, IdentityOp, LnOp, MulOp, NegOp, PowOp, RecipOp, SinOp,
    SqrtOp, SubOp, binary_elementwise, binary_elementwise_into, scalar_elementwise,
    scalar_elementwise_into, unary_elementwise, unary_elementwise_into,
};
pub use application::linalg::{
    batched_matmul, batched_matmul_into, det, dot, kron, kron_into, matexp, matmul, matmul_into,
    matpow, matrix_rank, matrix_rank_with_tolerance, norm_l1, norm_l2, norm_max, pinv, trace,
};
pub use application::reduction::{
    MaxOp, MinOp, PreparedAxisReduction, PreparedReduction, SumOp, max_axis, max_axis_into,
    mean_axis, mean_axis_into, min_axis, min_axis_into, prepare_max_axis_into,
    prepare_mean_axis_into, prepare_min_axis_into, prepare_reduce_axis_into, prepare_reduction,
    prepare_reduction_with_width, prepare_sum_axis_into, reduce_axis, reduction,
    reduction_with_width, submit_prepared_axis_reduction_batch, submit_prepared_reduction_batch,
    sum_axis, sum_axis_into,
};
pub use application::scan::{
    CumProdOp, CumSumOp, ScanDirection, cumsum, cumsum_into, scan_axis, scan_axis_into,
};
pub use application::stencil::{
    BoundaryCondition, Laplacian2DKernel, Laplacian2DParams, LaplacianPolarity,
};
pub use application::strided::{
    MAX_STRIDED_RANK, StridedOperand, binary_elementwise_strided, binary_elementwise_strided_into,
    scalar_elementwise_strided, scalar_elementwise_strided_into, unary_elementwise_strided,
    unary_elementwise_strided_into,
};
pub use application::volume::{
    FieldGeometry, RAY_STRIDE, ray_line_integrals, ray_line_integrals_into,
};
pub use infrastructure::buffer::MetalBuffer;
pub use infrastructure::device::MetalDevice;

#[cfg(feature = "decomposition")]
pub use application::decomposition::{
    GpuBidiagonalDecomposition, GpuBunchKaufmanDecomposition, GpuCholesky,
    GpuColPivQrDecomposition, GpuFullPivLuDecomposition, GpuHessenbergDecomposition,
    GpuLuDecomposition, GpuQrDecomposition, GpuRealSchur, GpuSvdDecomposition,
    GpuSymmetricEigenDecomposition, GpuUduDecomposition, bidiagonalize, bunch_kaufman,
    cholesky_decompose, cholesky_decompose_blocked, col_piv_qr, eigenvalues, full_piv_lu,
    hessenberg, lu_decompose, lu_decompose_blocked, qr_decompose, qr_decompose_blocked, schur,
    singular_values, svd_decompose, svd_rank_revealing, symmetric_eigen_jacobi,
    symmetric_eigenvalues_jacobi, udu_decompose,
};

pub use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};
