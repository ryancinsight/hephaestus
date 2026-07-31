use core::num::NonZeroUsize;

use crate::domain::view::StridedView;

/// Whether attention applies the autoregressive upper-triangular mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionCausality {
    /// Every query position may attend to every key position.
    Unrestricted,
    /// Query position `i` may attend only to key positions `j <= i`.
    Causal,
}

/// A grouped rank-2 keep mask over `[mask_batch, key_sequence]`.
///
/// `heads_per_batch` maps each contiguous group of execution batches to one
/// mask row. A mask with shape `[1, key_sequence]` therefore broadcasts by
/// setting this value to the complete execution-batch extent.
pub struct GroupedKeepMask<'a, B> {
    view: StridedView<'a, B, 2>,
    heads_per_batch: NonZeroUsize,
}

impl<'a, B> GroupedKeepMask<'a, B> {
    /// Create a grouped keep mask with a nonzero execution-batch group width.
    #[must_use]
    pub const fn new(view: StridedView<'a, B, 2>, heads_per_batch: NonZeroUsize) -> Self {
        Self {
            view,
            heads_per_batch,
        }
    }

    /// Borrow the mask buffer and layout.
    #[must_use]
    pub const fn view(&self) -> StridedView<'a, B, 2> {
        self.view
    }

    /// Number of consecutive execution batches represented by one mask row.
    #[must_use]
    pub const fn heads_per_batch(&self) -> NonZeroUsize {
        self.heads_per_batch
    }
}

impl<B> Clone for GroupedKeepMask<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Copy for GroupedKeepMask<'_, B> {}

/// Masking policy for accelerator scaled dot-product attention.
#[derive(Clone, Copy)]
pub struct AttentionMask<'a, B> {
    causality: AttentionCausality,
    keep: Option<GroupedKeepMask<'a, B>>,
}

impl<'a, B> AttentionMask<'a, B> {
    /// Create an unrestricted attention mask without a keep mask.
    #[must_use]
    pub const fn unrestricted() -> Self {
        Self {
            causality: AttentionCausality::Unrestricted,
            keep: None,
        }
    }

    /// Create a causal attention mask without a keep mask.
    #[must_use]
    pub const fn causal() -> Self {
        Self {
            causality: AttentionCausality::Causal,
            keep: None,
        }
    }

    /// Create an unrestricted attention mask with a grouped keep mask.
    #[must_use]
    pub const fn keep(keep: GroupedKeepMask<'a, B>) -> Self {
        Self {
            causality: AttentionCausality::Unrestricted,
            keep: Some(keep),
        }
    }

    /// Create a causal attention mask with a grouped keep mask.
    #[must_use]
    pub const fn causal_keep(keep: GroupedKeepMask<'a, B>) -> Self {
        Self {
            causality: AttentionCausality::Causal,
            keep: Some(keep),
        }
    }

    /// Return the causal policy.
    #[must_use]
    pub const fn causality(&self) -> AttentionCausality {
        self.causality
    }

    /// Return the optional grouped keep mask.
    #[must_use]
    pub const fn grouped_keep(&self) -> Option<GroupedKeepMask<'a, B>> {
        self.keep
    }
}
