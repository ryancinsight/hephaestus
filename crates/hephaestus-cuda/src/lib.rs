// The stub substrate (no `cuda` feature) performs no FFI and forbids unsafe.
// The real backend requires unsafe for the dynamically loaded driver/NVRTC
// FFI: it occurs in `infrastructure::{device, buffer, compiler}` and at the
// application-layer kernel-launch and device-copy sites (`pipeline`,
// `reduction`, and the `decomposition` modules, including their
// `bytemuck::Pod` metadata impls). Every unsafe block and impl carries a
// `// SAFETY:` note stating the invariants relied on.
#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
#![deny(missing_docs)]
//! # hephaestus-cuda
//!
//! CUDA backend for the Atlas shared accelerator substrate (atlas ADR 0001).
//! It is the GPU-side sibling of [`hephaestus-wgpu`]: it implements the same
//! [`hephaestus_core::ComputeDevice`] seam — device acquisition, a typed
//! [`CudaBuffer<T>`] device buffer, and host/device transfer — so consumers
//! that bind generically (`<D: ComputeDevice>`) substitute CUDA for wgpu
//! without source changes. Hephaestus is to the GPU what `leto` is to the CPU;
//! this crate adds the CUDA pathway alongside the portable wgpu one.
//!
//! The CUDA toolchain composes cuda-oxide for device acquisition, context
//! management, `CUdeviceptr` allocation, and host/device transfer with cutile
//! for kernel authoring above that substrate. Without the `cuda` feature, the
//! crate compiles as a stub whose [`CudaDevice::try_default`] reports the
//! backend unavailable rather than fabricating a device.
//!
//! [`hephaestus-wgpu`]: https://docs.rs/hephaestus-wgpu
//! [`CudaBuffer<T>`]: crate::CudaBuffer
//! [`CudaDevice::try_default`]: crate::CudaDevice::try_default
//!
//! ## Scope
//!
//! This crate currently owns the device substrate (acquisition, typed buffers,
//! transfer). Monomorphized elementwise/reduction kernel dispatch — mirroring
//! `hephaestus-wgpu`'s `application` layer — composes cutile PTX authoring on
//! top of this foundation and lands in a follow-up.

mod infrastructure;

/// Monomorphized CUDA compute dispatch.
pub mod application;

pub use application::elementwise::{
    AbsOp, AddOp, CosOp, DivOp, ExpNegOp, ExpOp, IdentityOp, LnOp, MulOp, NegOp, PowOp, RecipOp,
    SinOp, SqrtOp, SubOp, binary_elementwise, binary_elementwise_into, scalar_elementwise,
    scalar_elementwise_into, unary_elementwise, unary_elementwise_into,
};
#[cfg(feature = "decomposition")]
pub use application::linalg::MatrixDecompose;
pub use application::linalg::{
    AsGpuMatrixOperand, MatrixFunction, MatrixIdentityScalar, MatrixNorm, MatrixProduct,
    MatrixProperties, MatrixRankScalar, MatrixSolve, batched_matmul, batched_matmul_into, det, dot,
    kron, kron_into, matexp, matmul, matmul_into, matpow, matrix_rank, matrix_rank_with_tolerance,
    norm_l1, norm_l2, norm_max, pinv, trace,
};
pub use application::reduction::{
    MaxOp, MinOp, SumOp, max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis,
    min_axis_into, reduce_axis, reduce_axis_into, reduction, reduction_with_width, sum_axis,
    sum_axis_into,
};
pub use application::scan::{
    CumProdOp, CumSumOp, ScanDirection, cumsum, cumsum_into, scan_axis, scan_axis_into,
};
pub use application::storage_kernel::{CudaMultiStorageKernel, CudaStorageBinding};
pub use application::stream::{CudaCommandStream, CudaGroupedPrepared, CudaPrepared};
pub use application::strided::{
    MAX_STRIDED_RANK, StridedLayout, StridedOperand, StridedOperandDyn, binary_elementwise_strided,
    binary_elementwise_strided_dyn_into, binary_elementwise_strided_into,
    scalar_elementwise_strided, scalar_elementwise_strided_into, unary_elementwise_strided,
    unary_elementwise_strided_dyn_into, unary_elementwise_strided_into,
};

pub use application::random::{normal_with_seed, uniform_with_seed};
pub use application::sparse::{
    GpuCsrMatrix, spmm, spmm_into, spmv, spmv_into, spmv_many, spmv_many_into,
};
pub use infrastructure::buffer::CudaBuffer;
pub use infrastructure::device::CudaDevice;

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

pub use hephaestus_core::{
    BinaryExpr, BinaryStorageKernel, BlockWidth, CombineExpr, ComputeDevice,
    ComputeDeviceAcquisition, ComputeDeviceCapabilities, CudaC, DeviceBuffer, DeviceFeature,
    DeviceLimits, DialectScalar, GroupedBinding, GroupedCommandStream, GroupedKernelDevice,
    GroupedKernelInterface, GroupedKernelSource, HephaestusError, IdentityToken, KernelDevice,
    MultiStorageDevice, MultiStorageKernel, OpIdentity, Result, UnaryExpr, UnaryStorageKernel,
};
