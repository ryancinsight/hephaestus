//! The width-indexed dispatch seam for stateful updates.

use eunomia::{Pod, RealField};
use hephaestus_core::{
    ComputeDevice, HephaestusError, StatefulUpdateOperands, StatefulUpdateOps, StatefulUpdateRule,
    UpdateWidth,
};
use leto_ops::RealScalar;

/// Width-indexed stateful-update clause seam.
///
/// Extends the core [`UpdateWidth`] seam with the oracle bounds the clauses
/// need, so every clause below runs for both widths from one body instead of
/// a duplicated f64 entry.
pub trait StatefulUpdateWidth:
    UpdateWidth + RealField + RealScalar + Pod + PartialEq + core::fmt::Debug
{
    /// Dispatch one update at this width.
    fn dispatch<D, O, Rule, const N: usize>(
        device: &D,
        operations: &O,
        operands: StatefulUpdateOperands<'_, D::Buffer<Self>, N>,
        parameters: <Rule as StatefulUpdateRule<O::Dialect>>::Parameters<Self>,
    ) -> Result<(), HephaestusError>
    where
        D: ComputeDevice,
        O: StatefulUpdateOps<D>,
        Rule: StatefulUpdateRule<O::Dialect>,
    {
        operations.stateful_update::<Rule, Self, N>(device, operands, parameters)
    }
}

impl StatefulUpdateWidth for f32 {}

impl StatefulUpdateWidth for f64 {}
