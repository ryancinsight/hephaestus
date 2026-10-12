use super::super::device::ComputeDevice;
use super::super::dialect::KernelDialect;
use super::super::error::{HephaestusError, Result};
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

    /// Apply one f64 update in place to the parameter and persistent state.
    ///
    /// Backends override this once they ship f64 kernels; the default
    /// reports [`HephaestusError::Unsupported`] so f32-only backends need no
    /// stub and callers get a typed rejection instead of a dispatch failure.
    fn stateful_update_f64<Rule, const N: usize>(
        &self,
        _device: &D,
        _operands: StatefulUpdateOperands<'_, D::Buffer<f64>, N>,
        _parameters: <Rule as StatefulUpdateRule<Self::Dialect>>::ParametersF64,
    ) -> Result<()>
    where
        Rule: StatefulUpdateRule<Self::Dialect>,
    {
        Err(HephaestusError::Unsupported {
            message: format!(
                "{} does not support f64 stateful updates for {}",
                core::any::type_name::<D>(),
                core::any::type_name::<Rule>(),
            ),
        })
    }
}
