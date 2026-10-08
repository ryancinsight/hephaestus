use eunomia::Pod;

use super::CrossEntropyPlan;

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Portable scalar contract for accelerator cross-entropy.
///
/// The shared provider contract admits the scalars with native arithmetic on
/// every shipped device API plus conformance coverage. Additional scalar
/// types are admitted only when every provider has native arithmetic and
/// conformance coverage.
pub trait CrossEntropyScalar: private::Sealed + Pod + Copy + Send + Sync + 'static {
    /// Most negative finite value: the row-maximum seed mirroring the
    /// device preflight shaders (`-3.402823466e+38` / `-1.7976931348623157e308`).
    const FINITE_MIN: Self;

    /// Most positive finite value: the saturating upstream/destination
    /// fixture mirroring the device preflight shaders.
    const FINITE_MAX: Self;

    /// Width-native probability-row tolerance from a validated plan.
    #[must_use]
    fn tolerance(plan: &CrossEntropyPlan) -> Self;

    /// Width-native `upstream / batch` preflight scale, casting the batch
    /// count in native precision exactly as each width's kernels do.
    #[must_use]
    fn preflight_scale(upstream: Self, batch: usize) -> Self;
}

impl CrossEntropyScalar for f32 {
    const FINITE_MIN: Self = f32::MIN;
    const FINITE_MAX: Self = f32::MAX;

    fn tolerance(plan: &CrossEntropyPlan) -> Self {
        plan.probability_tolerance
    }

    fn preflight_scale(upstream: Self, batch: usize) -> Self {
        #[expect(
            clippy::cast_precision_loss,
            reason = "mirrors the WGSL shader's f32(dimensions.x) batch scale"
        )]
        let count = batch as f32;
        upstream / count
    }
}

impl CrossEntropyScalar for f64 {
    const FINITE_MIN: Self = f64::MIN;
    const FINITE_MAX: Self = f64::MAX;

    fn tolerance(plan: &CrossEntropyPlan) -> Self {
        plan.probability_tolerance_f64
    }

    fn preflight_scale(upstream: Self, batch: usize) -> Self {
        #[expect(
            clippy::cast_precision_loss,
            reason = "usize batch counts below 2^53 represent exactly in f64"
        )]
        let count = batch as f64;
        upstream / count
    }
}
