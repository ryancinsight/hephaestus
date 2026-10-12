#![expect(
    clippy::unwrap_used,
    reason = "ratchet HEPH-UNWRAP-1: pre-existing debt"
)]

//! Differential contract tests for CUDA strided-layout dispatch: device results vs
//! a CPU reference computed over the same leto layout metadata.

mod binary;
mod common;
mod forms;
mod unary;
