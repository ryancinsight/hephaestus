//! Device-neutral scaled dot-product attention operands and dispatch seam.

mod mask;
mod operands;
mod ops;
mod plan;
mod scalar;
mod status;
mod validation;

pub use mask::{AttentionCausality, AttentionMask, GroupedKeepMask};
pub use operands::{AttentionBackwardOperands, AttentionForwardOperands, AttentionGradientViews};
pub use ops::AttentionOps;
pub use plan::{AttentionPlan, plan_attention_backward, plan_attention_forward};
pub use scalar::AttentionScalar;
pub use status::AttentionSemanticStatus;
