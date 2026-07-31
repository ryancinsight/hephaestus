use crate::domain::view::StridedView;

use super::AttentionMask;

/// Borrowed operands for scaled dot-product attention forward.
pub struct AttentionForwardOperands<'a, B, T> {
    /// Query tensor in `[batch, query_sequence, key_feature]` order.
    pub query: StridedView<'a, B, 3>,
    /// Key tensor in `[batch, key_sequence, key_feature]` order.
    pub key: StridedView<'a, B, 3>,
    /// Value tensor in `[batch, key_sequence, value_feature]` order.
    pub value: StridedView<'a, B, 3>,
    /// Causal and optional grouped keep-mask policy.
    pub mask: AttentionMask<'a, B>,
    /// Runtime score scale.
    pub scale: T,
    /// Caller-owned output tensor in `[batch, query_sequence, value_feature]` order.
    pub output: StridedView<'a, B, 3>,
    /// Caller-owned post-softmax weights in `[batch, query_sequence, key_sequence]` order.
    pub weights: StridedView<'a, B, 3>,
}

/// Selected caller-owned destinations for additive attention backward.
pub struct AttentionGradientViews<'a, B> {
    /// Optional query-gradient destination.
    pub query: Option<StridedView<'a, B, 3>>,
    /// Optional key-gradient destination.
    pub key: Option<StridedView<'a, B, 3>>,
    /// Optional value-gradient destination.
    pub value: Option<StridedView<'a, B, 3>>,
}

impl<B> AttentionGradientViews<'_, B> {
    /// Return true when no gradient destination was selected.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.query.is_none() && self.key.is_none() && self.value.is_none()
    }
}

/// Borrowed operands for additive scaled dot-product attention backward.
pub struct AttentionBackwardOperands<'a, B, T> {
    /// Gradient of the forward output.
    pub grad_output: StridedView<'a, B, 3>,
    /// Forward query tensor.
    pub query: StridedView<'a, B, 3>,
    /// Forward key tensor.
    pub key: StridedView<'a, B, 3>,
    /// Forward value tensor.
    pub value: StridedView<'a, B, 3>,
    /// Stored post-softmax forward weights.
    pub weights: StridedView<'a, B, 3>,
    /// Runtime score scale used by the forward pass.
    pub scale: T,
    /// Selected additive gradient destinations.
    pub gradients: AttentionGradientViews<'a, B>,
}
