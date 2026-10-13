//! Metal implementation of the device-neutral decomposition seam (ADR 0042).
//!
//! Metal's decomposition machinery re-exports WGPU's handle types, which
//! implement the core handle traits only for `WgpuDevice`; the orphan rule
//! forecloses re-implementing them here for `MetalDevice`. Each result is
//! therefore wrapped in a thin newtype holding the WGPU handle plus a
//! rewrapped factor buffer (WGPU buffers clone as handles, so the rewrap is
//! a reference-count bump, not a copy).

mod basic;
mod ops;
mod orthogonal;
mod pivoted;
mod spectral;
mod symmetric;

pub use basic::{MetalCholeskyDecomposition, MetalLuDecomposition, MetalQrDecomposition};
pub use ops::MetalDecompositionOps;
pub use orthogonal::{MetalBidiagonalDecomposition, MetalHessenbergDecomposition, MetalRealSchur};
pub use pivoted::{MetalColPivQrDecomposition, MetalFullPivLuDecomposition};
pub use spectral::{MetalSvdDecomposition, MetalSymmetricEigenDecomposition};
pub use symmetric::{MetalBunchKaufmanDecomposition, MetalUduDecomposition};
