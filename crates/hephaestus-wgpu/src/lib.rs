#![deny(missing_docs)]
//! # hephaestus-wgpu
//!
//! The portable wgpu backend of the Atlas accelerator substrate (atlas ADR
//! 0001). Implements the `hephaestus-core` [`ComputeDevice`] seam over a
//! wgpu device/queue pair: adapter acquisition, typed device buffers, and
//! monomorphized elementwise compute dispatch.
//!
//! The crate re-exports the exact [`wgpu`] crate version it builds against so
//! downstream migration code can author provider-owned WGPU bindings without
//! adding a second direct `wgpu` dependency.
//!
//! [`ComputeDevice`]: hephaestus_core::ComputeDevice

/// Elementwise compute dispatch.
pub mod application;
/// wgpu device, queue, and buffer infrastructure.
pub mod infrastructure;

pub use application::elementwise::{
    binary_elementwise, binary_elementwise_into, scalar_elementwise, scalar_elementwise_into,
    unary_elementwise, unary_elementwise_into, AbsOp, AddOp, CosOp, DivOp, ExpNegOp, ExpOp,
    IdentityOp, LnOp, MulOp, NegOp, PowOp, RecipOp, SinOp, SqrtOp, SubOp,
};
#[cfg(feature = "decomposition")]
pub use application::linalg::MatrixDecompose;
pub use application::linalg::{
    batched_matmul, batched_matmul_into, det, dot, kron, kron_into, matexp, matmul, matmul_into,
    matpow, matrix_rank, matrix_rank_with_tolerance, norm_l1, norm_l2, norm_max, pinv, trace,
    AsGpuMatrixOperand, L2NormScalar, MatmulZero, MatrixFunction, MatrixIdentityScalar, MatrixNorm,
    MatrixProduct, MatrixProperties, MatrixRankScalar, MatrixSolve,
};
pub use application::random::{normal_with_seed, uniform_with_seed};
pub use application::reduction::{
    max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis, min_axis_into,
    prepare_max_axis_into, prepare_mean_axis_into, prepare_min_axis_into, prepare_reduce_axis_into,
    prepare_reduction, prepare_reduction_with_width, prepare_sum_axis_into, reduce_axis,
    reduce_axis_into, reduction, reduction_with_width, submit_prepared_axis_reduction_batch,
    submit_prepared_reduction_batch, sum_axis, sum_axis_into, MaxOp, MinOp, PreparedAxisReduction,
    PreparedReduction, SumOp,
};
pub use application::scan::{
    cumsum, cumsum_into, scan_axis, scan_axis_into, CumProdOp, CumSumOp, ScanDirection,
};
#[cfg(feature = "sparse")]
pub use application::sparse::{
    prepare_spmm, prepare_spmv, prepare_spmv_many, spmm, spmm_into, spmv, spmv_into, spmv_many,
    spmv_many_into, submit_prepared_sparse_batch, GpuCsrMatrix, PreparedSparseDispatch,
    PreparedSpmm, PreparedSpmv,
};
pub use application::storage_kernel::{
    WgslBinaryStorageKernel, WgslMultiStorageKernel, WgslStorageBinding, WgslStorageBindingLayout,
    WgslUnaryStorageKernel,
};
pub use application::stream::{WgpuCommandStream, WgpuGroupedPrepared, WgpuPrepared};
pub use application::strided::{
    binary_elementwise_strided, binary_elementwise_strided_into, scalar_elementwise_strided,
    scalar_elementwise_strided_into, unary_elementwise_strided, unary_elementwise_strided_into,
    StridedOperand, MAX_STRIDED_RANK,
};
pub use application::volume::{
    ray_line_integrals, ray_line_integrals_into, FieldGeometry, RAY_STRIDE,
};
pub use wgpu;

pub use infrastructure::buffer::WgpuBuffer;
pub use infrastructure::device::WgpuDevice;
pub use infrastructure::{StagingBufferGuard, UniformBufferGuard};

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

pub use hephaestus_core::{
    BinaryExpr, BinaryStorageKernel, CombineExpr, ComputeDevice, ComputeDeviceAcquisition,
    ComputeDeviceCapabilities, DeviceBuffer, DeviceFeature, DeviceLimits, DevicePreference,
    DialectScalar, DispatchGrid, GroupedBinding, GroupedCommandStream, GroupedKernelDevice,
    GroupedKernelInterface, GroupedKernelSource, HephaestusError, IdentityToken, KernelDialect,
    MultiStorageKernel, OpIdentity, Result, UnaryExpr, UnaryStorageKernel, Wgsl,
};
