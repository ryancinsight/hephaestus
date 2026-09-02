//! Native WGSL pooling and sliding-window operations.

mod metadata;
mod pooling;
mod prepared;
mod shader;
mod sliding_window;

pub use pooling::{PreparedPoolingBackward, PreparedPoolingForward, WgpuPoolingOps};
pub use sliding_window::{
    PreparedSlidingWindowFold, PreparedSlidingWindowUnfold, WgpuSlidingWindowOps,
};

#[cfg(test)]
mod tests;
