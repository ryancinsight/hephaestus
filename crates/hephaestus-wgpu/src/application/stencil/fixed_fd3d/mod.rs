//! Provider-owned fixed-scheme three-dimensional first derivatives.
//!
//! The device counterpart of `leto_ops::FiniteDifference3D`'s per-axis
//! sweeps. One WGSL source serves all five schemes behind a uniform branch,
//! so the families cannot drift apart in the way five separately written
//! stencils could. Like the staggered pair beside it the kernel is f32-only:
//! WGSL does not guarantee f64 storage, and a generic scalar would be a
//! falsely generic boundary.
//!
//! # Bit-exactness against the CPU path
//!
//! Each scheme's lane reproduces the provider's arithmetic in the same
//! operation order — the sixth-order interior keeps the exact left-associated
//! `((−9·f[i+2] + 45·f[i+1]) + −45·f[i−1]) …` chain, and the fourth-order
//! fall-back keeps `((−8·m1 + 8·p1) + −p2) + m2` — with the reciprocal scales
//! arriving precomputed in the parameter block. WGSL arithmetic is IEEE
//! without implicit contraction, so the same operations round the same way.
//! The grid covers the output domain: every scheme but the forward sweep
//! writes the input grid, while forward drops one plane on its axis and each
//! lane reads `f[c]` and `f[c+1]` from the input grid.

pub use hephaestus_core::{FixedFd3DParams, FixedFd3DScheme};

mod kernel;
mod ops;
mod source;

pub use kernel::FixedFd3DKernel;
pub use ops::WgpuFixedFd3DOps;

/// Scheme discriminants for cross-checking the uniform branch.
#[cfg(test)]
pub(crate) fn scheme_discriminants() -> [(FixedFd3DScheme, u32); 5] {
    use FixedFd3DScheme::*;
    [
        (CentralSecondOrder, 0),
        (CentralFourthOrder, 1),
        (CentralSixthOrder, 2),
        (StaggeredForward, 3),
        (StaggeredBackward, 4),
    ]
}

#[cfg(test)]
#[path = "../tests_fixed_fd3d.rs"]
mod tests;
