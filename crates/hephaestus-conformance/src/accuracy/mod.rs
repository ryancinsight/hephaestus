//! Per-operator accuracy sweeps over the failing ranges (ADR 0061 Decision 6,
//! HEPH-KERNEL-TAIL-ACCURACY).
//!
//! Each operator module below sweeps one rendering over its failing range plus
//! structural edges on every available backend, measuring distance from the
//! `f64` value-function reference in ULP of `f32`. The reference follows the
//! method documented on the value functions: for an `f32` input the defining
//! formula evaluated in `f64`, whose own rounding is about `2^-29` of an `f32`
//! ULP per operation.
//!
//! ## Bound discipline
//!
//! The numbers below are unchanged conformance targets. A native-builtin
//! allowance is an input to an error analysis, not a proof that its composition
//! meets the operator target:
//!
//! - C-family entries cite the CUDA Programming Guide Table 17 and HIP math
//!   API builtin rows, plus arithmetic costs (`0.5` ULP per correctly rounded
//!   `+ - *` under round-to-nearest, CUDA `/` at `2` ULP without `-prec-div`,
//!   which NVRTC does not set). Hardware evidence remains backend-specific.
//! - WGSL entries cite the concrete accuracy table: `/` at `2.5` ULP, `exp`
//!   at `3 + 2|x|` ULP, and `log` at `2^-21` absolute for `x` in `[0.5, 2]`
//!   and `3` ULP outside, together with its permitted arithmetic rounding and
//!   reassociation rules. The small-argument `exp`/`log` canary grades on the
//!   attached adapter are sampled evidence rather than stronger table bounds.
//!   The six allowed-intrinsic countermodels recorded in the accuracy design
//!   exceed operator targets while respecting individual builtin allowances.
//!   Neither the table entries nor the canary establish a portable bound for
//!   the full composition across every legal intrinsic/compiler result.
//!
//! Existing targets, input domains, mathematical operations and supported
//! backends remain unchanged. Explicit adapter/driver/compiler qualification
//! requires retained actual results and cohort identity plus independent review
//! and owner/coordinator acceptance. Unknown identity fields stay unknown. A
//! passing sweep provides evidence only for its observed configuration; no
//! sample proves a universal WGSL composition guarantee.
//!
//! A sweep that exceeds its target is a finding to investigate. It must not be
//! made green by increasing that target to fit the observation.
//!
//! ## Canary coverage note
//!
//! The `hephaestus-wgpu` canary probes `exp`/`log` at small arguments only,
//! so a form transparent to a builtin at large arguments (WGSL `expm1` past
//! `x = 80`, where the Vulkan driver spends 59 ULP) carries that grade in
//! its own bound instead of inheriting a canary grade the probes never
//! measured. Densifying the canary into a range-aware grade is follow-up
//! work; until then the per-op entries record the measured driver behavior
//! they absorb.
//!
//! ## Method
//!
//! [`SweepDialect`] selects the bound family; the backend test passes the
//! variant matching the dialect its seam authors (checked in debug builds, so
//! a miswired call fails loudly instead of asserting another dialect's
//! bound). [`crate::accuracy::sweep_inputs`] builds one dense range plus explicit edges and
//! the `±0`, `±inf`, `NaN` specials every value function defines; the two
//! `assert_*_sweep_within` runners dispatch once, download, and report the
//! worst input on failure.

mod celu;
mod elu;
mod erfc;
mod expm1;
mod gelu;
mod gelu_grad;
mod gelu_tanh;
mod gelu_tanh_grad;
mod log1p;
mod mish;
mod mish_grad;
mod sigmoid;
mod silu;
mod silu_grad;
mod softplus;
mod softplus_grad;

pub use celu::assert_celu_sweep;
pub use elu::assert_elu_sweep;
pub use erfc::assert_erfc_sweep;
pub use expm1::assert_expm1_sweep;
pub use gelu::assert_gelu_sweep;
pub use gelu_grad::assert_gelu_grad_sweep;
pub use gelu_tanh::assert_gelu_tanh_sweep;
pub use gelu_tanh_grad::assert_gelu_tanh_grad_sweep;
pub use log1p::assert_log1p_sweep;
pub use mish::assert_mish_sweep;
pub use mish_grad::assert_mish_grad_sweep;
pub use sigmoid::assert_sigmoid_sweep;
pub use silu::assert_silu_sweep;
pub use silu_grad::assert_silu_grad_sweep;
pub use softplus::assert_softplus_sweep;
pub use softplus_grad::assert_softplus_grad_sweep;

mod runner;

pub use runner::{
    SweepDialect, above_flush_floor, assert_parameterized_sweep_within,
    assert_unary_sweep_abs_within, assert_unary_sweep_within, below_cap, sweep_inputs,
    ulps_from_f64, without_subnormals,
};
