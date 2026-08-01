use super::super::device::ComputeDevice;
use super::super::dialect::KernelDialect;
use super::super::error::Result;
use super::operands::StatefulUpdateOperands;
use super::rules::StatefulUpdateRule;

/// Device-neutral stateful parameter-update dispatch.
///
/// `Rule` is a zero-sized marker. Static dispatch selects and monomorphizes the
/// complete kernel once per rule; no vtable or per-element rule branch exists.
pub trait StatefulUpdateOps<D: ComputeDevice> {
    /// Kernel dialect authored by this backend.
    type Dialect: KernelDialect;

    /// Apply one update in place to the parameter and persistent state.
    ///
    /// Validation completes before dispatch, so a rejected operation performs
    /// no mutation.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid parameters, state count, shape, storage
    /// span, writable aliasing, buffer aliasing, or backend dispatch failure.
    fn stateful_update<Rule, const N: usize>(
        &self,
        device: &D,
        operands: StatefulUpdateOperands<'_, D::Buffer<f32>, N>,
        parameters: <Rule as StatefulUpdateRule<Self::Dialect>>::Parameters,
    ) -> Result<()>
    where
        Rule: StatefulUpdateRule<Self::Dialect>;
}
