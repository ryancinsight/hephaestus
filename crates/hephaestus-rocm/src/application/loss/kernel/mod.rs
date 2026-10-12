mod backward;
mod forward;
mod prelude;

pub(super) use backward::{BACKWARD_ENTRY, BACKWARD_PREFLIGHT_ENTRY, backward_source};
pub(super) use forward::{
    FORWARD_ENTRY, FORWARD_MEAN_ENTRY, FORWARD_PREFLIGHT_ENTRY, forward_source,
};

use hephaestus_core::{DialectScalar, HipC};

/// Scalar spellings for the HIP C cross-entropy kernels: the element type
/// token plus the matching math builtins and unsuffixed/suffixed literals.
pub(super) trait HipCrossEntropyScalar: DialectScalar<HipC> {
    /// Exponential builtin (`expf` / `exp`).
    const EXP: &'static str;
    /// Natural-logarithm builtin (`logf` / `log`).
    const LOG: &'static str;
    /// Absolute-value builtin (`fabsf` / `fabs`).
    const FABS: &'static str;
    /// Maximum builtin (`fmaxf` / `fmax`).
    const FMAX: &'static str;
    /// Unit literal (`1.0f` / `1.0`).
    const ONE: &'static str;
    /// Zero literal (`0.0f` / `0.0`).
    const ZERO: &'static str;
}

impl HipCrossEntropyScalar for f32 {
    const EXP: &'static str = "expf";
    const LOG: &'static str = "logf";
    const FABS: &'static str = "fabsf";
    const FMAX: &'static str = "fmaxf";
    const ONE: &'static str = "1.0f";
    const ZERO: &'static str = "0.0f";
}

impl HipCrossEntropyScalar for f64 {
    const EXP: &'static str = "exp";
    const LOG: &'static str = "log";
    const FABS: &'static str = "fabs";
    const FMAX: &'static str = "fmax";
    const ONE: &'static str = "1.0";
    const ZERO: &'static str = "0.0";
}
