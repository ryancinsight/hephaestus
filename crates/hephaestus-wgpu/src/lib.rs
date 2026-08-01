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

pub use application::attention::WgpuAttentionOps;
pub use application::axis_reduction_seam::WgpuAxisReductionOps;
pub use application::convolution::WgpuConvolutionOps;
pub use application::decomposition_seam::WgpuDecompositionOps;
pub use application::dense_product_seam::WgpuDenseProductOps;
pub use application::elementwise::{
    AbsOp, AcosOp, AcoshOp, AddOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CeilOp, CosOp, CoshOp, DivOp,
    EluGradOp, EluOp, EqOp, ErfOp, ErfcOp, Exp2Op, ExpNegOp, ExpOp, Expm1Op, FloorOp, GeOp,
    GeluGradOp, GeluOp, GeluTanhGradOp, GeluTanhOp, GtOp, IdentityOp, LeOp, LgammaOp, LnOp,
    Log1pOp, Log2Op, Log10Op, LtOp, MishGradOp, MishOp, MulOp, NeOp, NegOp, PowOp, RecipOp,
    ReluGradOp, ReluOp, RoundOp, SigmoidGradOp, SigmoidOp, SignOp, SiluGradOp, SiluOp, SinOp,
    SinhOp, SoftplusGradOp, SoftplusOp, SqrtOp, SubOp, TanOp, TanhGradOp, TanhOp, TruncOp,
    binary_elementwise, binary_elementwise_into, binary_elementwise_typed,
    binary_elementwise_typed_into, scalar_elementwise, scalar_elementwise_into, unary_elementwise,
    unary_elementwise_into,
};
pub use application::elementwise_seam::{PreparedElementwise, WgpuElementwiseOps};
pub use application::full_reduction_seam::{PreparedFullReduction, WgpuFullReductionOps};
#[cfg(feature = "decomposition")]
pub use application::linalg::MatrixDecompose;
pub use application::linalg::{
    AsGpuMatrixOperand, L2NormScalar, MatmulZero, MatrixFunction, MatrixIdentityScalar, MatrixNorm,
    MatrixProduct, MatrixProperties, MatrixRankScalar, MatrixSolve, PreparedDot, PreparedL2Norm,
    batched_matmul, batched_matmul_into, det, dot, kron, kron_into, matmul, matmul_into, matpow,
    matrix_rank, matrix_rank_with_tolerance, norm_l1, norm_l2, norm_max, prepare_dot,
    prepare_norm_l2, trace,
};
#[cfg(any(feature = "decomposition", feature = "sparse"))]
pub use application::linalg::{matexp, pinv};
pub use application::parameterized_elementwise::{
    WgpuParameterizedUnaryOps, parameterized_unary_strided_into,
};
#[cfg(any(feature = "decomposition", feature = "sparse"))]
pub use application::random::{normal_with_seed, uniform_with_seed};
#[cfg(any(feature = "decomposition", feature = "sparse"))]
pub use application::random_seam::WgpuRandomOps;
pub use application::reduction::{
    MaxOp, MinOp, PreparedAxisReduction, PreparedReduction, ProdOp, SumOp, max_axis, max_axis_into,
    mean_axis, mean_axis_into, min_axis, min_axis_into, prepare_max_axis_into,
    prepare_mean_axis_into, prepare_min_axis_into, prepare_reduce_axis_into, prepare_reduction,
    prepare_reduction_with_width, prepare_sum_axis_into, prod_axis, prod_axis_into, reduce_axis,
    reduce_axis_into, reduction, reduction_with_width, submit_prepared_axis_reduction_batch,
    submit_prepared_mixed_reduction_batch, submit_prepared_reduction_batch, sum_axis,
    sum_axis_into,
};
pub use application::scan::{
    CumProdOp, CumSumOp, ScanDirection, cumprod, cumprod_into, cumsum, cumsum_into, scan_axis,
    scan_axis_into, suffix_prod, suffix_prod_into, suffix_sum, suffix_sum_into,
};
pub use application::scan_seam::{PreparedScan, WgpuScanOps};
#[cfg(feature = "sparse")]
pub use application::sparse::{
    GpuCsrMatrix, PreparedSparseDispatch, PreparedSpmm, PreparedSpmv, WgpuSparseOps, prepare_spmm,
    prepare_spmv, prepare_spmv_many, spmm, spmm_into, spmv, spmv_into, spmv_many, spmv_many_into,
    submit_prepared_sparse_batch,
};
pub use application::stencil::WgpuStencilOps;
pub use application::stencil::{
    BoundaryCondition, Laplacian2DKernel, Laplacian2DParams, LaplacianPolarity,
};
pub use application::storage_kernel::{
    WgslBinaryStorageKernel, WgslMultiStorageKernel, WgslStorageBinding, WgslStorageBindingLayout,
    WgslUnaryStorageKernel,
};
pub use application::stream::{WgpuCommandStream, WgpuGroupedPrepared, WgpuPrepared};
pub use application::strided::{
    MAX_STRIDED_RANK, StridedOperand, binary_elementwise_strided, binary_elementwise_strided_into,
    binary_elementwise_strided_typed, binary_elementwise_strided_typed_into,
    scalar_elementwise_strided, scalar_elementwise_strided_into, unary_elementwise_strided,
    unary_elementwise_strided_into,
};
pub use application::vector::{WgpuPreparedDot, WgpuPreparedNorm, WgpuVectorOps};
pub use application::volume::{
    FieldGeometry, RAY_STRIDE, WgpuRayIntegralOps, ray_line_integrals, ray_line_integrals_into,
};
pub use wgpu;

pub use infrastructure::buffer::WgpuBuffer;
pub use infrastructure::device::WgpuDevice;
pub use infrastructure::{StagingBufferGuard, UniformBufferGuard};

#[cfg(feature = "decomposition")]
pub use application::decomposition::{
    GpuBidiagonalDecomposition, GpuBunchKaufmanDecomposition, GpuCholesky,
    GpuColPivQrDecomposition, GpuFullPivLuDecomposition, GpuHessenbergDecomposition,
    GpuLuDecomposition, GpuQrDecomposition, GpuRealSchur, GpuSvdDecomposition,
    GpuSymmetricEigenDecomposition, GpuUduDecomposition, bidiagonalize, bunch_kaufman,
    cholesky_decompose, cholesky_decompose_blocked, col_piv_qr, col_piv_qr_blocked, eigenvalues,
    full_piv_lu, full_piv_lu_blocked, hessenberg, lu_decompose, lu_decompose_blocked, qr_decompose,
    qr_decompose_blocked, schur, singular_values, svd_decompose, svd_rank_revealing,
    symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi, udu_decompose,
};

pub use hephaestus_core::{
    BinaryExpr, BinaryStorageKernel, CombineExpr, ComputeDevice, ComputeDeviceAcquisition,
    ComputeDeviceCapabilities, DenseVectorOps, DeviceBuffer, DeviceFeature, DeviceLimits,
    DevicePreference, DialectScalar, DispatchGrid, GroupedBinding, GroupedCommandStream,
    GroupedKernelDevice, GroupedKernelInterface, GroupedKernelSource, HephaestusError,
    IdentityToken, KernelDialect, MultiStorageKernel, OpIdentity, Result, UnaryExpr,
    UnaryStorageKernel, Wgsl,
};
