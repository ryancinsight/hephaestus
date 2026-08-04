use crate::domain::view::StridedView;

/// Borrowed provider-resident operands for mean cross-entropy forward.
pub struct CrossEntropyForwardOperands<'a, B, I> {
    /// Logits in `[batch, classes]` order.
    pub logits: StridedView<'a, B, 2>,
    /// One `u32` target class per batch row.
    pub targets: StridedView<'a, I, 1>,
    /// Caller-owned scalar mean-loss destination with shape `[1]`.
    pub loss: StridedView<'a, B, 1>,
    /// Caller-owned normalized probabilities in `[batch, classes]` order.
    pub probabilities: StridedView<'a, B, 2>,
}

/// Borrowed provider-resident operands for additive cross-entropy backward.
pub struct CrossEntropyBackwardOperands<'a, B, I> {
    /// Provider-resident scalar gradient of the mean loss, with shape `[1]`.
    pub output_gradient: StridedView<'a, B, 1>,
    /// Saved normalized probabilities in `[batch, classes]` order.
    pub probabilities: StridedView<'a, B, 2>,
    /// One `u32` target class per batch row.
    pub targets: StridedView<'a, I, 1>,
    /// Caller-owned additive logit-gradient destination.
    pub logit_gradient: StridedView<'a, B, 2>,
}
