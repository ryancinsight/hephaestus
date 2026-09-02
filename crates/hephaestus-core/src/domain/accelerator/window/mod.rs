//! Generic pooling and sliding-window launches over runtime-compiled C-family devices.
//!
//! CUDA and HIP share one source generator and one host-side implementation.
//! The vendor crates provide only their DeviceApi mechanics and convert
//! WindowKey into their local pipeline key.

mod key;
mod launch;
mod metadata;
mod pooling;
mod prepared;
mod sliding;
mod source;

pub use key::{WindowKey, WindowOperation};
pub use pooling::CFamilyPoolingOps;
pub use prepared::{PreparedFold, PreparedPoolingBackward, PreparedPoolingForward, PreparedUnfold};
pub use sliding::CFamilySlidingWindowOps;
pub use source::{WindowDialect, c_family_window_source};
