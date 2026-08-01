use super::super::view::StridedView;

/// Borrowed parameter, gradient, and persistent-state views for one update.
///
/// Rules validate the state count before dispatch. The slice representation
/// keeps the public seam uniform while the rule marker still monomorphizes the
/// complete device kernel.
#[derive(Debug)]
pub struct StatefulUpdateOperands<'a, B, const N: usize> {
    /// Parameter storage updated in place.
    pub parameter: StridedView<'a, B, N>,
    /// Read-only gradient storage.
    pub gradient: StridedView<'a, B, N>,
    /// Persistent rule state updated in place.
    pub states: &'a [StridedView<'a, B, N>],
}

impl<B, const N: usize> Clone for StatefulUpdateOperands<'_, B, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B, const N: usize> Copy for StatefulUpdateOperands<'_, B, N> {}

/// Pairwise storage-alias facts established by a concrete backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StatefulUpdateAliasing {
    /// Parameter aliases the read-only gradient.
    pub parameter_gradient: bool,
    /// Parameter aliases state zero.
    pub parameter_state_zero: bool,
    /// Parameter aliases state one.
    pub parameter_state_one: bool,
    /// Gradient aliases state zero.
    pub gradient_state_zero: bool,
    /// Gradient aliases state one.
    pub gradient_state_one: bool,
    /// The two writable states alias each other.
    pub states: bool,
}

impl StatefulUpdateAliasing {
    pub(crate) const fn any(self, state_count: usize) -> bool {
        self.parameter_gradient
            || self.parameter_state_zero
            || self.gradient_state_zero
            || (state_count == 2
                && (self.parameter_state_one || self.gradient_state_one || self.states))
    }
}
