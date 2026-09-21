//! Runtime-parameter unary expressions and their device-neutral dispatch seam.
//!
//! Parameter values remain dispatch data. They never enter generated source or
//! pipeline-cache keys, so changing activation bounds reuses the compiled
//! kernel and changes only the two scalar arguments.

use eunomia::{NumericElement, RealField};
use leto::Layout;

use super::device::ComputeDevice;
use super::dialect::{CudaC, HipC, Host, KernelDialect, Wgsl};
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Unary expression over `x` and two runtime scalars named `first` and
/// `second` in dialect `L`.
pub trait ParameterizedUnaryExpr<L: KernelDialect>: Copy + Send + Sync + 'static {
    /// Expression mapping `x`, `first`, and `second` to one output value.
    const EXPR: &'static str;

    /// The operator applied to a value and its two runtime parameters, for a
    /// dialect that executes operators instead of rendering them (ADR 0061).
    ///
    /// `None` unless the operator carries [`ParameterizedUnaryValue`]: the
    /// [`Host`] blanket impl overrides this with
    /// [`ParameterizedUnaryValue::apply`], and an operator implementing this
    /// trait for the host without a value function reports `None`, which the
    /// host turns into a typed error. A seam impl can call this under the
    /// `Op: ParameterizedUnaryExpr<L>` bound it already receives. Real-valued
    /// only, mirroring [`UnaryExpr::value`](crate::UnaryExpr::value): every
    /// parameterized unary operator is a transcendental or comparison
    /// function with no integer definition (ADR 0061 Decision 2).
    #[must_use]
    fn value<T: RealField>(_x: T, _first: T, _second: T) -> Option<T> {
        None
    }
}

/// The parameterized unary operator as a value function, the operator's
/// definition (ADR 0061 Decision 6): real-valued, generic over every
/// [`RealField`] (`f32`/`f64` today), taking `x` and the two runtime scalars
/// `first`/`second` in the same roles [`ParameterizedUnaryExpr::EXPR`]
/// documents per operator.
pub trait ParameterizedUnaryValue: Copy + Send + Sync + 'static {
    /// `x` transformed by the operator under parameters `first` and `second`.
    fn apply<T: RealField>(x: T, first: T, second: T) -> T;
}

/// Every operator with a value function is a host parameterized unary op,
/// mirroring [`crate::UnaryValue`]'s [`Host`] blanket: a downstream
/// `impl ParameterizedUnaryExpr<Host>` without [`ParameterizedUnaryValue`]
/// stays legal and reports `None` from [`ParameterizedUnaryExpr::value`].
impl<Op: ParameterizedUnaryValue> ParameterizedUnaryExpr<Host> for Op {
    const EXPR: &'static str = "host";

    fn value<T: RealField>(x: T, first: T, second: T) -> Option<T> {
        Some(Op::apply(x, first, second))
    }
}

/// Device-neutral runtime-parameter unary operations over `f32` strided views.
///
/// The pair is operation-defined. Hardtanh interprets it as `(minimum,
/// maximum)` and Threshold as `(threshold, replacement)`. Gradient operations
/// ignore the unused second value while retaining one stable dispatch shape.
pub trait ParameterizedUnaryOps<D: ComputeDevice> {
    /// Kernel dialect authored by this backend.
    type Dialect: KernelDialect;

    /// Compute `output = Op(input, parameters)` elementwise.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch, an aliased output, a layout validation
    /// failure, or the backend dispatch failure.
    fn parameterized_unary_into<Op, const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<f32>, N>,
        parameters: [f32; 2],
        output: StridedView<'_, D::Buffer<f32>, N>,
    ) -> Result<()>
    where
        Op: ParameterizedUnaryExpr<Self::Dialect>;
}

/// Validate that a writable parameterized-unary layout is addressable and
/// injective, then return its logical element count.
///
/// Monotonically separable strides take an allocation-free proof path. Other
/// layouts use exact offset validation with temporary storage bounded by eight
/// bytes per logical element.
///
/// # Errors
///
/// Returns a dispatch error when the layout exceeds `storage_len`, its logical
/// size or writable span overflows, or distinct logical indices can alias.
pub fn validate_parameterized_output<const N: usize>(
    layout: &Layout<N>,
    storage_len: usize,
) -> Result<usize> {
    layout
        .validate_storage_len(storage_len)
        .map_err(layout_error)?;
    let len = layout.checked_size().map_err(layout_error)?;
    if len == 0 {
        return Ok(0);
    }

    if separable_nonoverlap(layout)? {
        return Ok(len);
    }
    if exact_nonoverlap(layout, len)? {
        Ok(len)
    } else {
        Err(nonoverlap_error())
    }
}

fn separable_nonoverlap<const N: usize>(layout: &Layout<N>) -> Result<bool> {
    let mut axes = [(0_usize, 0_usize); N];
    let mut active = 0;
    for (&extent, &stride) in layout.shape().iter().zip(&layout.strides()) {
        if extent <= 1 {
            continue;
        }
        let magnitude = stride.unsigned_abs();
        if magnitude == 0 {
            return Ok(false);
        }
        *axes
            .get_mut(active)
            .expect("invariant: active writable axes never exceed layout rank") =
            (magnitude, extent);
        active += 1;
    }
    let active_axes = axes
        .get_mut(..active)
        .expect("invariant: active writable axes never exceed layout rank");
    active_axes.sort_unstable_by_key(|&(stride, _)| stride);

    let mut covered_span = 1_usize;
    for &(stride, extent) in active_axes.iter() {
        if stride < covered_span {
            return Ok(false);
        }
        covered_span = (extent - 1)
            .checked_mul(stride)
            .and_then(|axis_span| covered_span.checked_add(axis_span))
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: "writable output layout span overflows".to_string(),
            })?;
    }
    Ok(true)
}

fn exact_nonoverlap<const N: usize>(layout: &Layout<N>, len: usize) -> Result<bool> {
    let (minimum, maximum) = layout.checked_min_max_offsets().map_err(layout_error)?;
    let span = maximum
        .checked_sub(minimum)
        .and_then(|distance| distance.checked_add(1))
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: "writable output layout span overflows".to_string(),
        })?;
    let words = span.checked_add(63).map(|bits| bits / 64).ok_or_else(|| {
        HephaestusError::DispatchFailed {
            message: "writable output layout bitset size overflows".to_string(),
        }
    })?;

    if words <= len {
        let mut occupied = Vec::new();
        occupied
            .try_reserve_exact(words)
            .map_err(allocation_error)?;
        occupied.resize(words, 0_u64);
        for_each_offset(layout, len, |offset| {
            let relative = offset - minimum;
            let word = relative / 64;
            let mask = 1_u64 << (relative % 64);
            let slot = occupied
                .get_mut(word)
                .expect("invariant: offset lies inside the validated physical span");
            if *slot & mask != 0 {
                return false;
            }
            *slot |= mask;
            true
        })
    } else {
        let mut offsets = Vec::new();
        offsets.try_reserve_exact(len).map_err(allocation_error)?;
        for_each_offset(layout, len, |offset| {
            offsets.push(offset);
            true
        })?;
        offsets.sort_unstable();
        Ok(offsets
            .windows(2)
            .all(|pair| matches!(pair, [left, right] if left != right)))
    }
}

fn for_each_offset<const N: usize>(
    layout: &Layout<N>,
    len: usize,
    mut visit: impl FnMut(usize) -> bool,
) -> Result<bool> {
    for linear in 0..len {
        let mut index = [0_usize; N];
        let mut remainder = linear;
        for (coordinate, &extent) in index.iter_mut().zip(&layout.shape()).rev() {
            *coordinate = remainder % extent;
            remainder /= extent;
        }
        let offset = layout.offset_of(index).map_err(layout_error)?;
        if !visit(offset) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn allocation_error(error: std::collections::TryReserveError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("writable output layout validation allocation failed: {error}"),
    }
}

fn layout_error(error: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("layout rejected: {error}"),
    }
}

fn nonoverlap_error() -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: "output layout must be non-overlapping".to_string(),
    }
}

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

// ── Value functions (ADR 0061) ──────────────────────────────────────────
//
// Every `apply` below reproduces its `Wgsl`/`CudaC`/`HipC` `EXPR` exactly,
// nested-`select`/ternary order included: `HardtanhOp` in particular is not
// a plain `clamp(x, first, second)`, because a reversed pair (`first >
// second`) makes the two forms diverge (`clamp` panics or reorders; the
// nested form does not — see the boundary tests below).

impl ParameterizedUnaryValue for HardtanhOp {
    fn apply<T: RealField>(x: T, first: T, second: T) -> T {
        let inner = if x > second { second } else { x };
        if x < first { first } else { inner }
    }
}

/// Open interval on both ends (ADR 0061): the boundary itself reports no
/// gradient, matching `(x > first) && (x < second)`.
impl ParameterizedUnaryValue for HardtanhGradOp {
    fn apply<T: RealField>(x: T, first: T, second: T) -> T {
        if x > first && x < second {
            <T as NumericElement>::ONE
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

/// Strict `x > first` selects `x`; the boundary and below select `second`,
/// matching `select(second, x, x > first)`.
impl ParameterizedUnaryValue for ThresholdOp {
    fn apply<T: RealField>(x: T, first: T, second: T) -> T {
        if x > first { x } else { second }
    }
}

/// Strict `x > first`, matching `select(0.0, 1.0, x > first)`.
impl ParameterizedUnaryValue for ThresholdGradOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if x > first {
            <T as NumericElement>::ONE
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

/// Non-strict `x >= 0` selects the identity branch, matching
/// `select(first * x, x, x >= 0.0)`; both signed zeros take the identity
/// branch.
impl ParameterizedUnaryValue for LeakyReluOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if x >= <T as NumericElement>::ZERO {
            x
        } else {
            first * x
        }
    }
}

/// Strict `x > 0.0`, matching `select(first, 1.0, x > 0.0)`: both signed
/// zeros select the negative-slope branch (unlike [`LeakyReluOp`]'s
/// non-strict forward pass).
impl ParameterizedUnaryValue for LeakyReluGradOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if x > <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE
        } else {
            first
        }
    }
}

/// Strict `abs(x) > first`, matching `select(0.0, x, abs(x) > first)`.
impl ParameterizedUnaryValue for HardshrinkOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if <T as NumericElement>::abs(x) > first {
            x
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

/// Strict `abs(x) > first`, matching `select(0.0, 1.0, abs(x) > first)`.
impl ParameterizedUnaryValue for HardshrinkGradOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if <T as NumericElement>::abs(x) > first {
            <T as NumericElement>::ONE
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

/// Strict on both branches, matching
/// `select(select(0.0, x - first, x > first), x + first, x < -first)`.
impl ParameterizedUnaryValue for SoftshrinkOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if x < -first {
            x + first
        } else if x > first {
            x - first
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

/// Strict on both branches, matching
/// `select(0.0, 1.0, (x > first) || (x < -first))`.
impl ParameterizedUnaryValue for SoftshrinkGradOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if x > first || x < -first {
            <T as NumericElement>::ONE
        } else {
            <T as NumericElement>::ZERO
        }
    }
}

/// `celu(x) = x` for `x ≥ 0`, `first · expm1(x / first)` for `x < 0` (`first`
/// is α) — `expm1` rather than `exp(x / first) - 1` keeps the negative
/// branch accurate near zero (ADR 0061 Decision 6, the same cancellation
/// class as [`crate::EluOp`]).
impl ParameterizedUnaryValue for CeluOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if x >= <T as NumericElement>::ZERO {
            x
        } else {
            first * (x / first).exp_m1()
        }
    }
}

/// Takes the input, matching the WGSL/CUDA rendering (not listed among the
/// forward-output gradients of ADR 0061 Decision 7).
impl ParameterizedUnaryValue for CeluGradOp {
    fn apply<T: RealField>(x: T, first: T, _second: T) -> T {
        if x >= <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE
        } else {
            (x / first).exp()
        }
    }
}

impl ParameterizedUnaryExpr<Wgsl> for HardtanhOp {
    const EXPR: &'static str = "select(select(x, second, x > second), first, x < first)";
}

impl ParameterizedUnaryExpr<Wgsl> for HardtanhGradOp {
    const EXPR: &'static str = "select(0.0, 1.0, (x > first) && (x < second))";
}

impl ParameterizedUnaryExpr<Wgsl> for ThresholdOp {
    const EXPR: &'static str = "select(second, x, x > first)";
}

impl ParameterizedUnaryExpr<Wgsl> for ThresholdGradOp {
    const EXPR: &'static str = "select(0.0, 1.0, x > first)";
}

impl ParameterizedUnaryExpr<Wgsl> for LeakyReluOp {
    const EXPR: &'static str = "select(first * x, x, x >= 0.0)";
}

impl ParameterizedUnaryExpr<Wgsl> for LeakyReluGradOp {
    const EXPR: &'static str = "select(first, 1.0, x > 0.0)";
}

impl ParameterizedUnaryExpr<Wgsl> for HardshrinkOp {
    const EXPR: &'static str = "select(0.0, x, abs(x) > first)";
}

impl ParameterizedUnaryExpr<Wgsl> for HardshrinkGradOp {
    const EXPR: &'static str = "select(0.0, 1.0, abs(x) > first)";
}

impl ParameterizedUnaryExpr<Wgsl> for SoftshrinkOp {
    const EXPR: &'static str = "select(select(0.0, x - first, x > first), x + first, x < -first)";
}

impl ParameterizedUnaryExpr<Wgsl> for SoftshrinkGradOp {
    const EXPR: &'static str = "select(0.0, 1.0, (x > first) || (x < -first))";
}

impl ParameterizedUnaryExpr<Wgsl> for CeluOp {
    const EXPR: &'static str = "select(first * (exp(x / first) - 1.0), x, x >= 0.0)";
}

impl ParameterizedUnaryExpr<Wgsl> for CeluGradOp {
    const EXPR: &'static str = "select(exp(x / first), 1.0, x >= 0.0)";
}

macro_rules! impl_c_family {
    ($dialect:ty) => {
        impl ParameterizedUnaryExpr<$dialect> for HardtanhOp {
            const EXPR: &'static str = "x < first ? first : (x > second ? second : x)";
        }

        impl ParameterizedUnaryExpr<$dialect> for HardtanhGradOp {
            const EXPR: &'static str = "(x > first && x < second) ? 1.0 : 0.0";
        }

        impl ParameterizedUnaryExpr<$dialect> for ThresholdOp {
            const EXPR: &'static str = "x > first ? x : second";
        }

        impl ParameterizedUnaryExpr<$dialect> for ThresholdGradOp {
            const EXPR: &'static str = "x > first ? 1.0 : 0.0";
        }

        impl ParameterizedUnaryExpr<$dialect> for LeakyReluOp {
            const EXPR: &'static str = "x >= 0.0f ? x : first * x";
        }

        impl ParameterizedUnaryExpr<$dialect> for LeakyReluGradOp {
            const EXPR: &'static str = "x > 0.0f ? 1.0f : first";
        }

        impl ParameterizedUnaryExpr<$dialect> for HardshrinkOp {
            const EXPR: &'static str = "fabsf(x) > first ? x : 0.0f";
        }

        impl ParameterizedUnaryExpr<$dialect> for HardshrinkGradOp {
            const EXPR: &'static str = "fabsf(x) > first ? 1.0f : 0.0f";
        }

        impl ParameterizedUnaryExpr<$dialect> for SoftshrinkOp {
            const EXPR: &'static str = "x > first ? x - first : (x < -first ? x + first : 0.0f)";
        }

        impl ParameterizedUnaryExpr<$dialect> for SoftshrinkGradOp {
            const EXPR: &'static str = "(x > first || x < -first) ? 1.0f : 0.0f";
        }

        impl ParameterizedUnaryExpr<$dialect> for CeluOp {
            const EXPR: &'static str = "x >= 0.0f ? x : first * (expf(x / first) - 1.0f)";
        }

        impl ParameterizedUnaryExpr<$dialect> for CeluGradOp {
            const EXPR: &'static str = "x >= 0.0f ? 1.0f : expf(x / first)";
        }
    };
}

impl_c_family!(CudaC);
impl_c_family!(HipC);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardtanh_keeps_the_open_lower_and_upper_boundary_at_x_itself() {
        // Boundary convention (ADR 0061): `select(select(x, second, x >
        // second), first, x < first)` uses strict `<`/`>`, so `x == first`
        // or `x == second` falls through both selects to `x` itself —
        // numerically identical to the clamped bound, but the code path
        // taken differs, which the reversed-bounds case below distinguishes.
        assert_eq!(HardtanhOp::apply(-0.75_f32, -0.75, 1.25), -0.75);
        assert_eq!(HardtanhOp::apply(1.25_f32, -0.75, 1.25), 1.25);
        assert_eq!(HardtanhOp::apply(-2.0_f32, -0.75, 1.25), -0.75);
        assert_eq!(HardtanhOp::apply(2.0_f32, -0.75, 1.25), 1.25);
    }

    #[test]
    fn hardtanh_matches_the_nested_select_order_under_reversed_bounds() {
        // A plain `x.clamp(first, second)` requires `first <= second` and
        // diverges from the nested-select rendering once `first > second`
        // (`clamp` panics in debug; a naive min/max reordering would give
        // the wrong branch here). `EXPR`'s outer select conditions on `x <
        // first` using the *original* `x`, not the inner select's result, so
        // working through `select(select(x, second, x > second), first, x <
        // first)` by hand for `first = 0.5 > second = -0.25`:
        // - x = 0.5: inner picks `second` (0.5 > -0.25); outer condition
        //   `x < first` is `0.5 < 0.5` = false, so the result is the inner
        //   value `second = -0.25`, not `first`.
        assert_eq!(HardtanhOp::apply(0.5_f32, 0.5, -0.25), -0.25);
        assert_eq!(HardtanhOp::apply(-2.0_f32, 0.5, -0.25), 0.5);
        assert_eq!(HardtanhOp::apply(2.0_f32, 0.5, -0.25), -0.25);
    }

    #[test]
    fn hardtanh_grad_is_the_open_interval_excluding_both_endpoints() {
        assert_eq!(HardtanhGradOp::apply(-0.75_f32, -0.75, 1.25), 0.0);
        assert_eq!(HardtanhGradOp::apply(1.25_f32, -0.75, 1.25), 0.0);
        assert_eq!(HardtanhGradOp::apply(0.0_f32, -0.75, 1.25), 1.0);
    }

    #[test]
    fn threshold_selects_the_replacement_at_and_below_the_boundary() {
        // `x > first ? x : second` (strict): `x == first` takes `second`.
        assert_eq!(ThresholdOp::apply(0.5_f32, 0.5, -3.25), -3.25);
        assert_eq!(ThresholdOp::apply(0.500_001_f32, 0.5, -3.25), 0.500_001);
    }

    #[test]
    fn threshold_grad_matches_the_strict_boundary() {
        assert_eq!(ThresholdGradOp::apply(0.5_f32, 0.5, 0.0), 0.0);
        assert_eq!(ThresholdGradOp::apply(0.500_001_f32, 0.5, 0.0), 1.0);
    }

    #[test]
    fn leaky_relu_takes_the_identity_branch_at_both_signed_zeros() {
        // `x >= 0.0` (non-strict): `-0.0 >= 0.0` is `true` in IEEE-754, so
        // both signed zeros take the identity branch, unlike the gradient.
        assert_eq!(LeakyReluOp::apply(0.0_f32, 0.25, 0.0), 0.0);
        assert_eq!(LeakyReluOp::apply(-0.0_f32, 0.25, 0.0), -0.0);
        assert_eq!(LeakyReluOp::apply(-1.0_f32, 0.25, 0.0), -0.25);
    }

    #[test]
    fn leaky_relu_grad_selects_the_negative_slope_at_both_signed_zeros() {
        // `x > 0.0` (strict): `-0.0 > 0.0` and `0.0 > 0.0` are both `false`,
        // so both signed zeros take the negative-slope branch — the exact
        // case `hephaestus-conformance`'s `assert_parameterized_unary_contract`
        // pins across every backend.
        assert_eq!(LeakyReluGradOp::apply(0.0_f32, 0.25, 0.0), 0.25);
        assert_eq!(LeakyReluGradOp::apply(-0.0_f32, 0.25, 0.0), 0.25);
        assert_eq!(LeakyReluGradOp::apply(1.0_f32, 0.25, 0.0), 1.0);
    }

    #[test]
    fn hardshrink_and_grad_use_the_strict_magnitude_threshold() {
        assert_eq!(HardshrinkOp::apply(0.5_f32, 0.5, 0.0), 0.0);
        assert_eq!(HardshrinkOp::apply(-0.5_f32, 0.5, 0.0), 0.0);
        assert_eq!(HardshrinkOp::apply(0.500_001_f32, 0.5, 0.0), 0.500_001);
        assert_eq!(HardshrinkGradOp::apply(0.5_f32, 0.5, 0.0), 0.0);
        assert_eq!(HardshrinkGradOp::apply(0.500_001_f32, 0.5, 0.0), 1.0);
    }

    #[test]
    fn softshrink_and_grad_use_the_strict_threshold_on_both_sides() {
        assert_eq!(SoftshrinkOp::apply(0.5_f32, 0.5, 0.0), 0.0);
        assert_eq!(SoftshrinkOp::apply(-0.5_f32, 0.5, 0.0), 0.0);
        // Compared against `x - first`/`x + first` computed the same way the
        // implementation computes it, not a separately-rounded literal: the
        // two nearby f32 values subtract exactly (Sterbenz's lemma), but an
        // independently parsed `0.000_001_f32` literal rounds to a different
        // nearest float than that exact difference.
        let just_above = 0.500_001_f32;
        assert_eq!(SoftshrinkOp::apply(just_above, 0.5, 0.0), just_above - 0.5);
        let just_below = -0.500_001_f32;
        assert_eq!(SoftshrinkOp::apply(just_below, 0.5, 0.0), just_below + 0.5);
        assert_eq!(SoftshrinkGradOp::apply(0.5_f32, 0.5, 0.0), 0.0);
        assert_eq!(SoftshrinkGradOp::apply(0.500_001_f32, 0.5, 0.0), 1.0);
        assert_eq!(SoftshrinkGradOp::apply(-0.500_001_f32, 0.5, 0.0), 1.0);
    }

    #[test]
    fn celu_uses_exp_m1_to_avoid_negative_tail_cancellation() {
        // celu(x) = alpha * expm1(x / alpha) for x < 0. At x = -1e-8, alpha
        // = 1, exp(-1e-8) rounds to exactly 1.0 in f32 (1e-8 is below
        // f32::EPSILON ~= 1.19e-7), so a direct `exp(x) - 1` transcription
        // cancels to 0.0 exactly, losing the tail entirely (ADR 0061
        // Decision 6). `exp_m1` computes it to a few ULP of the true value,
        // which is `x` to first order (the `x^2/2` term is ~5e-17,
        // negligible at f32 precision) — bound derived as 8 ULP at this
        // magnitude for eunomia's `exp_m1` accuracy plus rounding headroom.
        let value = CeluOp::apply(-1.0e-8_f32, 1.0, 0.0);
        assert!(value != 0.0, "celu(-1e-8) must not cancel to zero");
        let tolerance = 8.0 * f32::EPSILON * 1.0e-8_f32;
        assert!(
            (value - (-1.0e-8_f32)).abs() <= tolerance,
            "celu(-1e-8) = {value}, expected ~-1e-8 within {tolerance}"
        );
    }

    #[test]
    fn celu_matches_x_at_and_above_zero_and_the_exp_branch_below() {
        assert_eq!(CeluOp::apply(0.0_f32, 0.5, 0.0), 0.0);
        assert_eq!(CeluOp::apply(2.0_f32, 0.5, 0.0), 2.0);
        assert_eq!(CeluGradOp::apply(0.0_f32, 0.5, 0.0), 1.0);
        // Compared against `(x / first).exp()` via the same `RealField::exp`
        // the implementation calls (eunomia's `libm`-routed `exp`, not
        // necessarily bit-identical to `std::f32::exp`).
        assert_eq!(
            CeluGradOp::apply(-1.0_f32, 0.5, 0.0),
            (-1.0_f32 / 0.5).exp()
        );
    }

    #[test]
    fn value_functions_are_generic_over_every_shipped_real_field() {
        // Generic Instantiation Coverage (standards): one check function,
        // instantiated across every `RealField` eunomia ships today
        // (f32/f64), rather than a per-type test copy.
        fn check<T: RealField>() {
            let two = T::from_f64(2.0);
            let neg_one = T::from_f64(-1.0);
            let one = T::from_f64(1.0);
            let half = T::from_f64(0.5);
            let zero = <T as NumericElement>::ZERO;

            assert_eq!(HardtanhOp::apply(two, neg_one, one), one);
            assert_eq!(HardtanhOp::apply(-two, neg_one, one), neg_one);
            assert_eq!(ThresholdOp::apply(two, one, neg_one), two);
            assert_eq!(ThresholdOp::apply(zero, one, neg_one), neg_one);
            assert_eq!(LeakyReluOp::apply(-two, half, zero), -two * half);
            assert_eq!(HardshrinkOp::apply(-two, one, zero), -two);
            assert_eq!(HardshrinkOp::apply(half, one, zero), zero);
            assert_eq!(SoftshrinkOp::apply(two, one, zero), one);
            assert_eq!(CeluOp::apply(two, one, zero), two);
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn expressions_pin_parameter_and_boundary_conventions() {
        assert_eq!(
            <HardtanhOp as ParameterizedUnaryExpr<Wgsl>>::EXPR,
            "select(select(x, second, x > second), first, x < first)"
        );
        assert_eq!(
            <HardtanhGradOp as ParameterizedUnaryExpr<Wgsl>>::EXPR,
            "select(0.0, 1.0, (x > first) && (x < second))"
        );
        assert_eq!(
            <ThresholdOp as ParameterizedUnaryExpr<CudaC>>::EXPR,
            "x > first ? x : second"
        );
        assert_eq!(
            <ThresholdGradOp as ParameterizedUnaryExpr<HipC>>::EXPR,
            "x > first ? 1.0 : 0.0"
        );
        assert_eq!(
            <LeakyReluOp as ParameterizedUnaryExpr<Wgsl>>::EXPR,
            "select(first * x, x, x >= 0.0)"
        );
        assert_eq!(
            <LeakyReluGradOp as ParameterizedUnaryExpr<Wgsl>>::EXPR,
            "select(first, 1.0, x > 0.0)"
        );
        assert_eq!(
            <LeakyReluGradOp as ParameterizedUnaryExpr<CudaC>>::EXPR,
            "x > 0.0f ? 1.0f : first"
        );
        assert_eq!(
            <LeakyReluGradOp as ParameterizedUnaryExpr<HipC>>::EXPR,
            "x > 0.0f ? 1.0f : first"
        );
        assert_eq!(
            <SoftshrinkOp as ParameterizedUnaryExpr<CudaC>>::EXPR,
            "x > first ? x - first : (x < -first ? x + first : 0.0f)"
        );
        assert_eq!(
            <CeluGradOp as ParameterizedUnaryExpr<HipC>>::EXPR,
            "x >= 0.0f ? 1.0f : expf(x / first)"
        );
    }

    #[test]
    fn writable_layout_validation_accepts_injective_interleaving() {
        let layout = Layout::try_new([2, 3], [3, 2], 0).expect("valid test layout");
        let len = validate_parameterized_output(&layout, 8).expect("injective layout");
        assert_eq!(len, 6);
    }

    #[test]
    fn writable_layout_validation_rejects_nonzero_stride_aliasing() {
        let layout = Layout::try_new([2, 2], [1, 1], 0).expect("valid test layout");
        assert!(matches!(
            validate_parameterized_output(&layout, 3),
            Err(HephaestusError::DispatchFailed { message })
                if message == "output layout must be non-overlapping"
        ));
    }

    #[test]
    fn writable_layout_validation_uses_bounded_sparse_fallback() {
        let injective = Layout::try_new([2, 3], [300, 200], 0).expect("valid test layout");
        let len = validate_parameterized_output(&injective, 701).expect("injective sparse layout");
        assert_eq!(len, 6);

        let overlapping = Layout::try_new([2, 3], [300, 150], 0).expect("valid test layout");
        assert!(matches!(
            validate_parameterized_output(&overlapping, 601),
            Err(HephaestusError::DispatchFailed { message })
                if message == "output layout must be non-overlapping"
        ));
    }
}
