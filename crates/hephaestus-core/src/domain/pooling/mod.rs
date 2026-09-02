//! Device-neutral pooling contracts and plans.

mod operands;
mod ops;
mod plan;

pub use operands::{PoolingBackwardOperands, PoolingForwardOperands};
pub use ops::{PoolingMode, PoolingOps};
pub use plan::{PoolingPlan, plan_pooling_backward, plan_pooling_forward};
