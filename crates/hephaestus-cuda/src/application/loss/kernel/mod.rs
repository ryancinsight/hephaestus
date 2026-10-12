mod backward;
mod forward;
mod preflight;
mod prelude;

pub(super) use backward::backward_source;
pub(super) use forward::{forward_mean_source, forward_source};
pub(super) use preflight::{backward_preflight_source, forward_preflight_source};

use hephaestus_core::{CudaC, DialectScalar};

/// Scalar spellings for the CUDA C cross-entropy kernels: the element type
/// token plus the matching math builtins and unsuffixed/suffixed literals.
pub(super) trait CudaCrossEntropyScalar: DialectScalar<CudaC> {
    /// Exponential builtin (`expf` / `exp`).
    const EXP: &'static str;
    /// Natural-logarithm builtin (`logf` / `log`).
    const LOG: &'static str;
    /// Absolute-value builtin (`fabsf` / `fabs`).
    const FABS: &'static str;
    /// Unit literal (`1.0f` / `1.0`).
    const ONE: &'static str;
    /// Zero literal (`0.0f` / `0.0`).
    const ZERO: &'static str;
}

impl CudaCrossEntropyScalar for f32 {
    const EXP: &'static str = "expf";
    const LOG: &'static str = "logf";
    const FABS: &'static str = "fabsf";
    const ONE: &'static str = "1.0f";
    const ZERO: &'static str = "0.0f";
}

impl CudaCrossEntropyScalar for f64 {
    const EXP: &'static str = "exp";
    const LOG: &'static str = "log";
    const FABS: &'static str = "fabs";
    const ONE: &'static str = "1.0";
    const ZERO: &'static str = "0.0";
}
