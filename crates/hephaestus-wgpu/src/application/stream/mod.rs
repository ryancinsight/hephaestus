#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "ratchet HEPH-UNWRAP-1: pre-existing test-only debt"
    )
)]

//! WGPU implementation of the backend-neutral kernel command stream.

mod bound;
mod command;
mod commands;
mod device;
mod grouped;
mod kernel;
mod prepared;
mod storage;
mod storage_bindings;

#[cfg(test)]
mod tests;

pub use bound::WgpuBoundGroupedDispatch;
pub(crate) use command::WgpuBoundDispatch;
pub use command::WgpuCommandStream;
pub use grouped::WgpuGroupedSequence;
pub use prepared::{WgpuGroupedPrepared, WgpuPrepared};
