#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "ratchet HEPH-UNWRAP-1: pre-existing test-only debt"
    )
)]

//! Generic WGSL storage-kernel dispatch.

mod binary;
mod bindings;
mod multi;
mod unary;

pub use binary::WgslBinaryStorageKernel;
pub use bindings::{WgslStorageAccess, WgslStorageBinding, WgslStorageBindingLayout};
pub use multi::WgslMultiStorageKernel;
#[cfg(test)]
pub(crate) use multi::validate_distinct_bindings;
pub use unary::WgslUnaryStorageKernel;

#[cfg(test)]
#[path = "../tests_storage_kernel.rs"]
mod tests;
