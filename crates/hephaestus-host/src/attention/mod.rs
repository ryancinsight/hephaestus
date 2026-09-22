//! Leto as a scaled dot-product attention seam implementor (ADR 0046).
//!
//! [`HostAttentionOps`] mirrors the read-only preflight contract
//! `hephaestus-wgpu`'s `application/attention` module shares across its
//! forward and backward kernels: every triggered
//! [`AttentionSemanticStatus`](hephaestus_core::AttentionSemanticStatus)
//! reduces to its numerically lowest code, exactly as the WGSL kernels
//! combine per-invocation failures through `atomicMin` on one shared status
//! word, and only a `Valid` preflight reaches leto-ops'
//! `scaled_dot_product_attention_into` /
//! `scaled_dot_product_attention_backward_accumulate`, the only code paths
//! that touch a destination buffer.
//!
//! Forward's preflight is order-independent by construction: `atomicMin`
//! commutes, so the host recomputes every WGSL preflight kernel's status
//! (`query`/`key`/`value`/`keep`-mask finiteness, then weight-arithmetic
//! finiteness) and folds them with [`u32::min`] rather than short-circuiting.
//! Backward mirrors the same five independent finiteness kernels plus the
//! probability-row check, the score-gradient workspace finiteness check, and
//! one destination/arithmetic pair per requested gradient target — again
//! folded rather than short-circuited, since leto-ops' own sequential
//! `validate_backward` checks `grad_output` before `query`/`key` (codes 5
//! before 1/2), which would pick the wrong winner whenever both fail.

mod backward;
mod forward;
mod read;
mod seam;

pub use seam::{HostAttentionBackward, HostAttentionForward, HostAttentionOps};
