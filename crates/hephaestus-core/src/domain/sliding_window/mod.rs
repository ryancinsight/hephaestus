//! Device-neutral unfold/fold contracts and plans.

mod operands;
mod ops;
mod plan;

pub use operands::{SlidingWindowFoldOperands, SlidingWindowUnfoldOperands};
pub use ops::SlidingWindowOps;
pub use plan::{SlidingWindowPlan, plan_sliding_window_fold, plan_sliding_window_unfold};
