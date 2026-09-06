//! CUDA device ownership, acquisition, capabilities, and transfers.

mod access;
mod acquisition;
mod capabilities;
mod compute;
mod context;
mod properties;
mod state;

pub(crate) use context::{CudaContext, CurrentContext};
pub use state::CudaDevice;
