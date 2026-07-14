//! Linear algebra operations on the CUDA device.
//!
//! Two families with distinct dispatch strategies share this module:
//!
//! - `matmul` / `batched_matmul` (the `matmul` submodule): a bespoke tiled GPU
//!   kernel, authored as CUDA C and launched directly via `cuLaunchKernel`.
//! - `dot` / `trace` / `norm_l1` / `norm_l2` / `norm_max` (the `norms`
//!   submodule): compositions of the elementwise and reduction primitives over
//!   strided views — no bespoke kernel, so they inherit every backend
//!   optimization of those primitives.

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{HephaestusError, Result};
use leto::Layout;

mod kron;
mod matmul;
mod matpow;
mod matrix;
mod matrix_rank;
mod norms;
mod pinv_matexp;

pub use kron::{kron, kron_into};
pub use matmul::{batched_matmul, batched_matmul_into, matmul, matmul_into};
pub use matpow::{MatrixIdentityScalar, matpow};
#[cfg(feature = "decomposition")]
pub use matrix::MatrixDecompose;
pub use matrix::{
    AsGpuMatrixOperand, MatrixFunction, MatrixNorm, MatrixProduct, MatrixProperties, MatrixSolve,
};
pub use matrix_rank::{MatrixRankScalar, det, matrix_rank, matrix_rank_with_tolerance};
pub use norms::{dot, norm_l1, norm_l2, norm_max, trace};
pub use pinv_matexp::{matexp, pinv};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GpuMatrixLayout {
    pub(super) shape: [u32; 2],
    pub(super) strides: [i32; 2],
    pub(super) offset: u32,
    pub(super) _pad: [u32; 3],
}

#[inline]
pub(super) fn map_layout(layout: &Layout<2>) -> Result<GpuMatrixLayout> {
    Ok(GpuMatrixLayout {
        shape: [
            to_u32(layout.shape[0], "dimension")?,
            to_u32(layout.shape[1], "dimension")?,
        ],
        strides: [
            to_i32(layout.strides[0], "stride")?,
            to_i32(layout.strides[1], "stride")?,
        ],
        offset: to_u32(layout.offset, "offset")?,
        _pad: [0; 3],
    })
}

/// Map a leto layout-validation error into a dispatch failure.
///
/// Shared by both families: layout validation precedes every kernel launch and
/// every composed reduction, and a rejected layout is a dispatch contract
/// violation rather than a device error.
#[inline]
pub(super) fn map_layout_err(e: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("layout rejected: {e}"),
    }
}

/// Convert a `usize` extent to the `u32` the device-side layout struct uses.
#[inline]
pub(crate) fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds u32 range"),
    })
}

/// Convert an `isize` stride to the `i32` the device-side layout struct uses.
#[inline]
pub(crate) fn to_i32(value: isize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i32 range"),
    })
}

/// Convert an `isize` batch stride to the `long long` (`i64`) a batched
/// kernel argument uses. Unlike the per-matrix `i32` strides, a batch stride
/// is multiplied by up to the batch count on the device side, so it needs
/// the wider range even though `isize` is losslessly representable on every
/// 64-bit target this crate builds for.
#[inline]
pub(crate) fn to_i64(value: isize, what: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i64 range"),
    })
}
