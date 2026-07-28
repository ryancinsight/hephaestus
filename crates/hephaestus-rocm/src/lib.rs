#![cfg_attr(not(all(feature = "rocm", target_os = "linux")), forbid(unsafe_code))]
#![deny(missing_docs)]

//! # hephaestus-rocm
//!
//! Native AMD ROCm/HIP device substrate for the Atlas accelerator stack.
//!
//! The `rocm` feature enables the Linux HIP runtime implementation. Without
//! that feature, [`RocmDevice::try_default`] returns a typed unavailable-device
//! error and the crate remains buildable on hosts without ROCm. The backend
//! implements the shared [`hephaestus_core::ComputeDevice`] seam for device
//! acquisition, typed device buffers, host/device transfers, and
//! synchronization, and hipRTC/module-launched elementwise, reduction,
//! rank-2 axis-reduction, scan, map-reduction, Kronecker-product,
//! matrix-power, matrix-multiplication, and CSR sparse matrix products
//! operation families. Additional operator families are separate parity
//! increments with their own value-semantic contracts. The optional
//! `decomposition` feature adds device-resident Cholesky, LU, and QR
//! factorization contracts, including complete-pivoted LU, column-pivoted QR,
//! bidiagonalization, SVD, UDU, Bunch–Kaufman, Hessenberg, real Schur, and
//! eigenvalue decompositions.
//!
//! [`hephaestus_core::ComputeDevice`]: hephaestus_core::ComputeDevice

#[cfg(all(feature = "rocm", not(target_os = "linux")))]
compile_error!("the hephaestus-rocm `rocm` feature requires a Linux ROCm installation");

mod infrastructure;

/// Runtime-compiled ROCm compute operations.
pub mod application;

pub use infrastructure::{RocmBuffer, RocmDevice};

pub use application::axis_reduction::{
    max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis, min_axis_into, prod_axis,
    prod_axis_into, reduce_axis, reduce_axis_into, sum_axis, sum_axis_into,
};
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
#[cfg(feature = "decomposition")]
pub use application::linalg::MatrixDecompose;
pub use application::linalg::{
    AsGpuMatrixOperand, L2NormScalar, MatrixFunction, MatrixIdentityScalar, MatrixNorm,
    MatrixProduct, MatrixProperties, MatrixRankScalar, MatrixSolve, batched_matmul,
    batched_matmul_into, det, dot, kron, kron_into, matmul, matmul_into, matpow, matrix_rank,
    matrix_rank_with_tolerance, norm_l1, norm_l2, norm_max, trace,
};
pub use application::linalg::{matexp, pinv};
pub use application::prepared_axis_reduction::{
    PreparedAxisReduction, prepare_max_axis_into, prepare_mean_axis_into, prepare_min_axis_into,
    prepare_reduce_axis_into, prepare_sum_axis_into, submit_prepared_axis_reduction_batch,
};
pub use application::prepared_map_reduction::{
    PreparedDot, PreparedL2Norm, prepare_dot, prepare_norm_l2,
};
pub use application::prepared_reduction::{
    PreparedReduction, prepare_reduction, prepare_reduction_with_width,
    submit_prepared_reduction_batch,
};
pub use application::random::{normal_with_seed, uniform_with_seed};
pub use application::reduction::{MaxOp, MinOp, SumOp, reduction, reduction_with_width};
pub use application::scan::{
    CumProdOp, CumSumOp, ScanDirection, cumprod, cumprod_into, cumsum, cumsum_into, scan_axis,
    scan_axis_into, suffix_prod, suffix_prod_into, suffix_sum, suffix_sum_into,
};
pub use application::sparse::{
    GpuCsrMatrix, PreparedSparseDispatch, PreparedSpmm, PreparedSpmv, prepare_spmm, prepare_spmv,
    prepare_spmv_many, spmm, spmm_into, spmv, spmv_into, spmv_many, spmv_many_into,
    submit_prepared_sparse_batch,
};
pub use application::stencil::{
    BoundaryCondition, Laplacian2DKernel, Laplacian2DParams, LaplacianPolarity,
};
pub use application::storage_kernel::{RocmMultiStorageKernel, RocmStorageBinding};
pub use application::stream::{RocmCommandStream, RocmGroupedPrepared, RocmPrepared};
pub use application::strided::StridedOperand;
pub use application::strided_elementwise::{
    MAX_STRIDED_RANK, binary_elementwise_strided, binary_elementwise_strided_into,
    binary_elementwise_strided_typed, binary_elementwise_strided_typed_into,
    scalar_elementwise_strided, scalar_elementwise_strided_into, unary_elementwise_strided,
    unary_elementwise_strided_into,
};
pub use application::vector::{
    PreparedDot as RocmPreparedDot, PreparedL2Norm as RocmPreparedNorm, RocmVectorOps,
};
pub use application::volume::{
    FieldGeometry, RAY_STRIDE, ray_line_integrals, ray_line_integrals_into,
};

pub use hephaestus_core::{
    ComputeDevice, ComputeDeviceAcquisition, ComputeDeviceCapabilities, DenseVectorOps,
    DeviceBuffer, DeviceFeature, DeviceLimits, DevicePreference, HephaestusError, Result,
};
