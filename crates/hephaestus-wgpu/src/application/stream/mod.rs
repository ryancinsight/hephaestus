#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "ratchet HEPH-UNWRAP-1: pre-existing test-only debt"
    )
)]
//! WGPU implementation of the backend-neutral kernel command stream.

mod bound;
mod dispatch;
mod prepared;
mod sequence;
mod storage;
#[cfg(test)]
mod tests;

pub use bound::WgpuBoundGroupedDispatch;
pub use prepared::{WgpuGroupedPrepared, WgpuPrepared};
pub(crate) use sequence::WgpuBoundDispatch;
pub use sequence::WgpuCommandStream;
pub use sequence::WgpuGroupedSequence;
