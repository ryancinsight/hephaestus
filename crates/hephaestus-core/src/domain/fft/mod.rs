//! Device-neutral dense complex Fourier-transform planning and dispatch.
//!
//! The contract uses split real and imaginary device buffers so backends can
//! preserve lane-friendly storage without imposing a complex-number ABI. Ranks
//! one through three share one const-generic planner and operation seam.

mod capacity;
mod operands;
mod ops;
mod plan;

pub use capacity::{FftLimit, FftResource};
pub use operands::FftOperands;
pub use ops::FftOps;
pub use plan::{FftDirection, FftPlan, plan_fft, plan_fft_axes, plan_fft_shape};
