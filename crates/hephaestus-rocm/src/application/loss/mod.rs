//! Native HIP mean cross-entropy.

mod kernel;
mod metadata;
mod prepared;
mod resources;
mod seam;

pub use prepared::{PreparedRocmCrossEntropyBackward, PreparedRocmCrossEntropyForward};
pub use seam::RocmCrossEntropyOps;
