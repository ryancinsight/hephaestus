use super::super::device::ComputeDevice;
use super::super::dialect::KernelDialect;
use super::super::error::Result;
use super::operands::StatefulUpdateOperands;
use super::rules::StatefulUpdateRule;
use super::scalar::StatefulScalar;

/// Device-neutral stateful parameter-update dispatch.
///
/// `Rule` is a zero-sized marker and `T` a monomorphized width. Static dispatch
/// selects and compiles the complete kernel once per `(T, Rule)` pair; no
/// vtable or per-element rule branch exists.
pub trait StatefulUpdateOps<D: ComputeDevice> {
    /// Kernel dialect authored by this backend.
    type Dialect: KernelDialect;

    /// Apply one update in place to the parameter and persistent state.
    ///
    /// Validation completes before dispatch, so a rejected operation performs
    /// no mutation. Backends reject a width they do not run with
    /// [`HephaestusError::Unsupported`](crate::HephaestusError::Unsupported)
    /// before any mutation.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid parameters, state count, shape, storage
    /// span, writable aliasing, buffer aliasing, unsupported width, or backend
    /// dispatch failure.
    fn stateful_update<T, Rule, const N: usize>(
        &self,
        device: &D,
        operands: StatefulUpdateOperands<'_, D::Buffer<T>, N>,
        parameters: <Rule as StatefulUpdateRule<Self::Dialect>>::Parameters<T>,
    ) -> Result<()>
    where
        T: StatefulScalar,
        Rule: StatefulUpdateRule<Self::Dialect>;
}
