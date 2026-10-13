//! Backend-neutral connectionist temporal classification.
//!
//! # What this is for
//!
//! The Graves forward-backward loss over `[frames, batch, classes]`
//! log-probabilities. Leto owns it on the CPU as `leto_ops::CtcState`; this
//! is the device side of the same contract, so one loss reaches either
//! backend.
//!
//! # Why steps, and why two traits
//!
//! The frame recurrence is sequential — frame `t` reads frame `t − 1` — so no
//! single dispatch can span time. The seam therefore exposes one step per
//! frame and time step (`alpha`, `beta`), a per-sample likelihood finish,
//! and the embarrassingly parallel posterior, and a single shared driver in
//! this module walks them. A backend that ran its own loop would duplicate
//! the walk five times over; a host that faked steps it never takes would be
//! a mock wearing a trait impl. So [`CtcStepOps`](crate::domain::ctc::CtcStepOps) is the step seam GPUs bind,
//! [`CtcOps`](crate::domain::ctc::CtcOps) is the coarse seam every backend — host included — binds, and
//! GPU coarse methods are one call into the shared driver.
//!
//! # Log-space arithmetic
//!
//! Messages are `Weight` pairs `(high, low)` in the provider's compensated
//! form: TwoSum addition, log-sum-exp merge through `ln(1 + exp(d))`, and
//! `-infinity` for unreachable paths. Kernels reproduce the provider's
//! operation order exactly, including the large/small selection tie-break,
//! so CPU and device agree up to elementary-function rounding. Buffers hold
//! two `f32` lanes per message; the second lane of each pair is the residual.
//!
//! # What the driver keeps on the device
//!
//! [`CtcStateBuffers`](crate::domain::ctc::CtcStateBuffers) owns the forward recurrences between the passes —
//! alpha and beta, the expanded labels, per-sample metadata, likelihoods,
//! and divisors — allocated once per problem and retained across backward,
//! exactly the provider state's device image. The scalar loss is summed on
//! the host from the per-sample likelihoods in sample order, mirroring the
//! provider's accumulation and its overflow checks.
//!
//! # Preconditions the device does not check
//!
//! Active log-probabilities must be nonpositive finite-or-`-infinity`
//! lanes — log-softmax outputs satisfy this — the compensated intermediates
//! must stay finite, which holds for every representable problem, and
//! backward must not run when forward reported a non-finite loss: the
//! provider errors on impossible alignments, while a kernel can only write
//! the zeros the posterior takes there. Callers check the returned loss
//! first; the conformance suite pins the forward infinities and skips those
//! backward passes. The upstream seed, by contrast, is checked on the host
//! where the check is free.

mod kernel;
mod ops;
mod params;
mod problem;
mod state;

#[cfg(test)]
mod tests;

pub use kernel::CtcKernel;
pub use ops::{CtcOps, CtcStepOps, drive_ctc_forward};
pub use params::{CtcBackwardParams, CtcStepParams};
pub use problem::{CtcProblem, CtcSampleMeta};
pub use state::CtcStateBuffers;
