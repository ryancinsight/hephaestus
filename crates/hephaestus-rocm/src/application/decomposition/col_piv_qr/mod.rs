//! HIP column-pivoted Householder QR factorization.
//!
//! Each ordered HIP step selects the largest remaining column norm, swaps that
//! column into the active position, applies one Householder reflector to the
//! working matrix, and accumulates **Q**. The packed **R**, materialized **Q**,
//! column permutation, and rank are device-backed. The retained host factor
//! exists only for the established least-squares contract shared by CUDA and
//! wgpu.

mod api;
mod source;
mod stages;
mod types;

pub use api::{col_piv_qr, col_piv_qr_blocked};
pub use types::GpuColPivQrDecomposition;

#[cfg(test)]
pub(crate) use source::kernel_source;

#[cfg(test)]
mod tests;
