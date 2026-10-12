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
pub use unary::WgslUnaryStorageKernel;
