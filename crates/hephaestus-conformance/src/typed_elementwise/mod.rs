//! Contract clauses for scalar-aware (typed) binary elementwise dispatch.
//!
//! Each clause is generic over the device and the elementwise seam, so every
//! backend runs the same assertions. Comparisons are the only operators with
//! per-scalar-type codegen, so `u32`, `i32`, and `f32` are each instantiated
//! rather than one standing in for the others, and every oracle is an exact
//! equality: comparisons produce indicator values, the float operands are
//! finite dyadic values (with the `-0.0 == 0.0` identity and an adjacent pair
//! at the `2^24` exact-integer limit), and the discriminating signed case is
//! negative operands, which a kernel reusing the unsigned expression would
//! order incorrectly. NaN and infinity operands are deliberately absent: WGSL
//! implementations may assume they do not occur, so their behaviour is
//! capability-gated, not a universal clause.

mod backend;
mod comparisons;
mod entry;
mod helpers;
mod reuse;
mod shapes;
mod special;

pub use backend::TypedElementwiseBackend;
pub use entry::assert_typed_elementwise_contract;
