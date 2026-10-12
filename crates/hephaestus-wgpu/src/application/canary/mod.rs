//! Per-builtin driver capability canary for WGSL.
//!
//! Spec-valid WGSL compiles and runs exactly on most drivers, but three f64
//! transcendental builtins are driver-defective on the recorded NVIDIA Vulkan
//! adapter (RTX 5080, vendor `0x10DE`, device `0x2C02`): `log` and `tanh`
//! abort pipeline compilation (`0xc0000005` / `0x80000003`, bare-wgpu repro
//! with zero repo code on the stack), and `pow` passes validation then
//! access-violates at dispatch. A fourth, `exp`, dispatches but computes at
//! f32 grade (`e^1` equals `f32(e)` bit-for-bit). The templates are spec-valid,
//! so a blanket static gate would wrongly disable good drivers; the full fix
//! is an out-of-process dynamic probe (a crash-class compile cannot be probed
//! in-process without aborting the parent). Until then this module pairs two
//! mechanisms behind one query surface:
//!
//! - [`builtin_verdict`](crate::application::canary::builtin_verdict): a recorded table keyed by exact adapter identity
//!   (backend, vendor, device). Known-bad triples report [`BuiltinVerdict`](crate::application::canary::BuiltinVerdict);
//!   every unknown adapter (or device without adapter info) default-allows,
//!   preserving current behavior where nothing is proven.
//! - [`probe_precision`](crate::application::canary::probe_precision): a live one-dispatch measurement of a builtin's
//!   precision grade against CPU truth. It refuses crash-class pairs up front
//!   (via the table) so probing itself can never abort the process.
//!
//! Seams gate crash-class dispatch with [`require_expr_safe`](crate::application::canary::require_expr_safe) (which scans the
//! op template text for builtin calls, so composite expressions like
//! `GeluTanhOp` are covered without a manual op-to-builtin map) or the
//! explicit [`require_builtin`](crate::application::canary::require_builtin), turning process aborts into typed
//! [`HephaestusError::Unsupported`](hephaestus_core::HephaestusError::Unsupported) naming the cause.

mod probe;
mod scan;
mod table;
#[cfg(test)]
mod tests;
mod widths;

pub use probe::{PrecisionGrade, PrecisionReport, probe_precision};
pub use scan::{require_expr_safe, scan_builtins};
pub use table::{BuiltinVerdict, builtin_verdict, require_builtin};
pub use widths::{CanaryWidth, MathBuiltin, width_of};
