//! Scalar widths for precision-generic stateful updates.
//!
//! [`UpdateWidth`] is the single seam over which rule parameters, value
//! updates, and backend dispatches genericize: one body per rule, one
//! method per operation, monomorphized per width instead of twinned
//! per-width items.

use eunomia::{FloatElement, Pod, Zeroable};

/// Scalar widths a stateful-update rule runs at.
///
/// The bound set mirrors what the generic rule bodies need: [`Pod`] for
/// device buffers, [`FloatElement`] for arithmetic, domain checks,
/// bias-correction powers, and fixture lifts, and [`core::fmt::Display`]
/// for invalid-value diagnostics. The trait is open: a future width
/// implements it (plus its padding rule below) and every generic body
/// monomorphizes for it.
pub trait UpdateWidth: Pod + FloatElement + core::fmt::Display {
    /// Uniform padding cells for a rule block needing `N` f32 pad lanes.
    ///
    /// f32 keeps the frozen uniform ABI (`[f32; N]`, 16/32-byte blocks);
    /// f64 field bytes already satisfy granularity, so its pad is empty.
    type Pad<const N: usize>: Pod + Zeroable;
}

impl UpdateWidth for f32 {
    type Pad<const N: usize> = [f32; N];
}

impl UpdateWidth for f64 {
    type Pad<const N: usize> = [f64; 0];
}
