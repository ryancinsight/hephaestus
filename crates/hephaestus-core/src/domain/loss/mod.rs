//! Device-neutral mean cross-entropy operands and dispatch seam.

mod operands;
mod ops;
mod plan;
mod scalar;
mod status;

pub use operands::{CrossEntropyBackwardOperands, CrossEntropyForwardOperands};
pub use ops::CrossEntropyOps;
pub use plan::{CrossEntropyPlan, plan_cross_entropy_backward, plan_cross_entropy_forward};
pub use scalar::CrossEntropyScalar;
pub use status::CrossEntropyStatus;
