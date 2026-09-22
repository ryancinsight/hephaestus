//! The runtime-parameter operator markers.
//!
//! Each is a zero-sized type naming one operation; the pair of runtime
//! scalars it reads is documented per marker and rendered by the dialect
//! expressions beside it.

/// Hardtanh `clamp(x, minimum, maximum)` marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardtanhOp;

/// Hardtanh open-interval derivative marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardtanhGradOp;

/// Threshold replacement marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ThresholdOp;

/// Threshold strict-greater-than derivative marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ThresholdGradOp;

/// Leaky ReLU activation marker; `first` is the negative slope.
#[derive(Clone, Copy, Debug, Default)]
pub struct LeakyReluOp;

/// Leaky ReLU gradient marker; `first` is the negative slope, including at
/// both signed zeros.
#[derive(Clone, Copy, Debug, Default)]
pub struct LeakyReluGradOp;

/// Hardshrink activation marker; `first` is the threshold.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardshrinkOp;

/// Hardshrink gradient marker; `first` is the threshold.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardshrinkGradOp;

/// Softshrink activation marker; `first` is the threshold.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftshrinkOp;

/// Softshrink gradient marker; `first` is the threshold.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftshrinkGradOp;

/// Continuously differentiable exponential linear unit marker; `first` is α.
#[derive(Clone, Copy, Debug, Default)]
pub struct CeluOp;

/// Continuously differentiable exponential linear unit gradient marker; `first` is α.
#[derive(Clone, Copy, Debug, Default)]
pub struct CeluGradOp;
