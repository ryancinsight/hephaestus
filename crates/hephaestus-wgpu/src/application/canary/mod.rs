//! Per-builtin capability canary for WGSL math builtins.
//!
//! naga lowers WGSL `exp`, `log`, `pow`, and `tanh` on the Vulkan backend to
//! the GLSL.std.450 extended instructions `Exp` (27), `Log` (28), `Pow` (26),
//! and `Tanh` (21) at every float width, but the GLSL.std.450 specification
//! admits only 16- or 32-bit floating-point operands for those instructions
//! (<https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html>).
//! An f64 operand is invalid SPIR-V, so every Vulkan driver's behavior on it
//! is undefined. The observed behaviors span the full range: on NVIDIA
//! RTX 5080 (driver 610.60) f64 `log`/`tanh` abort pipeline compilation, f64
//! `pow` access-violates at dispatch, and f64 `exp` computes at f32 grade; on
//! Mesa lavapipe (hosted CI) f64 `exp` reaches a worst relative error of 1. The
//! defect is the lowering, not an adapter, so the verdict is keyed by
//! backend and width rather than by adapter identity. Two mechanisms share
//! one query surface:
//!
//! - [`builtin_verdict`](crate::application::canary::builtin_verdict): the
//!   specification rule. f64 transcendentals on Vulkan report
//!   [`BuiltinVerdict::Blocked`](crate::application::canary::BuiltinVerdict)
//!   with the cause; every other pair, and a device without adapter info,
//!   reports supported.
//! - [`probe_precision`](crate::application::canary::probe_precision): a live
//!   one-dispatch measurement of a builtin's precision grade against CPU
//!   truth. It refuses blocked pairs before building any shader, so probing
//!   can never compile undefined SPIR-V or abort the process.
//!
//! Seams gate dispatch with [`require_expr_safe`](crate::application::canary::require_expr_safe)
//! (which scans the WGSL text for builtin calls, so composite expressions
//! like `GeluTanhOp` are covered without a manual op-to-builtin map) or the
//! explicit [`require_builtin`](crate::application::canary::require_builtin),
//! turning undefined results and process aborts into a typed
//! [`HephaestusError::Unsupported`](hephaestus_core::HephaestusError::Unsupported)
//! naming the cause.

mod accuracy;
mod probe;
mod scan;
mod verdict;

pub use probe::{PrecisionGrade, PrecisionReport, probe_precision};
pub(crate) use scan::require_template_safe;
#[cfg(test)]
pub(crate) use scan::{RESTRICTED_TOKENS, calls, token_width};
pub use scan::{require_expr_safe, scan_builtins};
pub use verdict::{
    BuiltinVerdict, CanaryWidth, MathBuiltin, builtin_verdict, require_builtin, width_of,
};

#[cfg(test)]
mod tests;
