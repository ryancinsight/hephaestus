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
//! Every bound is derived from its form's published error source, never tuned
//! to the measurement:
//!
//! - C-family renderings that call one native builtin cite that builtin's
//!   vendor table row (CUDA Programming Guide Table 17, HIP math API), plus
//!   `0.5` ULP per correctly-rounded `+ - *` and the table row for `/`
//!   (CUDA `2` without `-prec-div`, which NVRTC does not set).
//! - WGSL renderings cite the WGSL floating-point accuracy table as
//!   implemented by the WebGPU CTS (`+ - *` correctly rounded, `/` at `2.5`
//!   ULP, `exp` at `3 + 2|x|` ULP, `log` at `2^-21` absolute for `x` in
//!   `[0.5, 2]` and `3` ULP outside), plus the live canary grade the
//!   `hephaestus-wgpu` canary probes on the attached adapter: forms that need
//!   `log` near `1` to behave relatively (where the spec floor is absolute)
//!   are sound only where the canary holds, which the sweep then verifies per
//!   backend rather than assumes.
//!
//! A sweep that exceeds its bound is a finding, not a tuning prompt:
//! investigate the form, the builtin, or the reference, and widen a bound
//! only with the cause recorded at the bound.
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
