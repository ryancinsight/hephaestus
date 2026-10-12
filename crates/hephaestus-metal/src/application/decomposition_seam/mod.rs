//! Metal implementation of the device-neutral decomposition seam (ADR 0042).
//!
//! Metal's decomposition machinery re-exports WGPU's handle types, which
//! implement the core handle traits only for `WgpuDevice`; the orphan rule
//! forecloses re-implementing them here for `MetalDevice`. Each result is
//! therefore wrapped in a thin newtype holding the WGPU handle plus a
//! rewrapped factor buffer (WGPU buffers clone as handles, so the rewrap is
//! a reference-count bump, not a copy).

mod handles;
mod ops;

pub use handles::{
    MetalBidiagonalDecomposition, MetalBunchKaufmanDecomposition, MetalCholeskyDecomposition,
    MetalColPivQrDecomposition, MetalFullPivLuDecomposition, MetalHessenbergDecomposition,
    MetalLuDecomposition, MetalQrDecomposition, MetalRealSchur, MetalSvdDecomposition,
    MetalSymmetricEigenDecomposition, MetalUduDecomposition,
};
pub use ops::MetalDecompositionOps;
