//! Owned CUDA driver dispatch. CUDAAPI is `extern "system"`; C `size_t`
//! arguments and outputs are `usize`, including on Windows LLP64 targets.
//! Symbols resolve once at acquisition, never inside transfer or kernel loops.

mod attributes;
mod context;
mod device;
mod kernel;
mod library;
mod loader;
mod memory;
mod region;

pub(crate) use attributes::Attribute;
pub(crate) use loader::Driver;
pub(crate) use region::{CopyRegion, Endpoint};

#[cfg(test)]
mod tests;

#[cfg(not(target_pointer_width = "64"))]
compile_error!("hephaestus-cuda requires the 64-bit CUDA driver ABI");
