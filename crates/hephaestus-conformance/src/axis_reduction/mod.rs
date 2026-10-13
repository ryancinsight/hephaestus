//! Contract clauses for device-neutral rank-2 axis reductions.
//!
//! Each clause is generic over the device and the seam, so every backend runs
//! the same assertions. The fixture is the 3x4 matrix `[[1,2,3,4],[5,6,7,8],
//! [9,10,11,12]]`: small enough to state expected values inline, and chosen so
//! every product stays below `2^24` and is therefore exact in `f32`.

mod backend;
mod entry;
mod fixture;
mod mean;
mod minmax;
mod prepared;
mod prod;
mod sum;

pub use backend::ReductionBackend;
pub use entry::assert_axis_reduction_contract;
