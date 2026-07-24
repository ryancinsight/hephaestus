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
//! bidiagonalization, SVD, UDU, and Bunch–Kaufman.
//!
//! [`hephaestus_core::ComputeDevice`]: hephaestus_core::ComputeDevice

#[cfg(all(feature = "rocm", not(target_os = "linux")))]
compile_error!("the hephaestus-rocm `rocm` feature requires a Linux ROCm installation");

mod infrastructure;

/// Runtime-compiled ROCm compute operations.
pub mod application;

pub use infrastructure::{RocmBuffer, RocmDevice};

pub use application::axis_reduction::{
    max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis, min_axis_into, reduce_axis,
    reduce_axis_into, sum_axis, sum_axis_into,
};
#[cfg(feature = "decomposition")]
pub use application::decomposition::{
    GpuBidiagonalDecomposition, GpuBunchKaufmanDecomposition, GpuCholesky,
    GpuColPivQrDecomposition, GpuFullPivLuDecomposition, GpuLuDecomposition, GpuQrDecomposition,
    GpuSvdDecomposition, GpuUduDecomposition, bidiagonalize, bunch_kaufman, cholesky_decompose,
    cholesky_decompose_blocked, col_piv_qr, col_piv_qr_blocked, full_piv_lu, full_piv_lu_blocked,
    lu_decompose, lu_decompose_blocked, qr_decompose, qr_decompose_blocked, singular_values,
    svd_decompose, svd_rank_revealing, udu_decompose,
};
pub use application::elementwise::{
    binary_elementwise, binary_elementwise_into, scalar_elementwise, scalar_elementwise_into,
    unary_elementwise, unary_elementwise_into,
};
pub use application::linalg::{
    L2NormScalar, MatrixIdentityScalar, MatrixRankScalar, batched_matmul, batched_matmul_into, det,
    dot, kron, kron_into, matmul, matmul_into, matpow, matrix_rank, matrix_rank_with_tolerance,
    norm_l1, norm_l2, norm_max, trace,
};
pub use application::random::{normal_with_seed, uniform_with_seed};
pub use application::reduction::{MaxOp, MinOp, SumOp, reduction, reduction_with_width};
pub use application::scan::{
    CumProdOp, CumSumOp, ScanDirection, cumprod, cumprod_into, cumsum, cumsum_into, scan_axis,
    scan_axis_into,
};
pub use application::sparse::{
    GpuCsrMatrix, spmm, spmm_into, spmv, spmv_into, spmv_many, spmv_many_into,
};
pub use application::storage_kernel::{RocmMultiStorageKernel, RocmStorageBinding};
pub use application::stream::{RocmCommandStream, RocmGroupedPrepared, RocmPrepared};
pub use application::strided::StridedOperand;
pub use application::strided_elementwise::{
    MAX_STRIDED_RANK, binary_elementwise_strided, binary_elementwise_strided_into,
    scalar_elementwise_strided, scalar_elementwise_strided_into, unary_elementwise_strided,
    unary_elementwise_strided_into,
};

pub use hephaestus_core::{
    ComputeDevice, ComputeDeviceAcquisition, ComputeDeviceCapabilities, DeviceBuffer,
    DeviceFeature, DeviceLimits, DevicePreference, HephaestusError, Result,
};
