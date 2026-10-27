//! Sweep runners and input builders shared by the per-operator accuracy modules.

use eunomia::FloatElement;
use hephaestus_core::{
    ComputeDevice, CudaC, DialectScalar, ElementwiseOps, HipC, ParameterizedUnaryExpr,
    ParameterizedUnaryOps, ParameterizedUnaryValue, StridedView, UnaryExpr, UnaryValue, Wgsl,
};
use leto::Layout;

// ──SweepDialect & input filters ───────────────────────────────────────────────

/// Which published bound family a sweep asserts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepDialect {
    /// CUDA C library bounds (CUDA Programming Guide Table 17).
    CudaC,
    /// HIP C library bounds (HIP math API reference).
    HipC,
    /// WGSL accuracy-table bounds plus the live canary grade.
    Wgsl,
}

impl SweepDialect {
    /// Fail loudly when a sweep is wired to another dialect's bound.
    pub(crate) fn check<Dialect: hephaestus_core::KernelDialect>(self) {
        let expected = core::any::TypeId::of::<Dialect>();
        let actual = match self {
            SweepDialect::CudaC => core::any::TypeId::of::<CudaC>(),
            SweepDialect::HipC => core::any::TypeId::of::<HipC>(),
            SweepDialect::Wgsl => core::any::TypeId::of::<Wgsl>(),
        };
        debug_assert_eq!(
            expected, actual,
            "accuracy sweep wired to the wrong dialect bound"
        );
    }
}

/// Drop subnormal inputs, keeping `±0`, normals, infinities, and NaN (for WGSL).
#[must_use]
pub fn without_subnormals(inputs: &[f32]) -> Vec<f32> {
    inputs
        .iter()
        .copied()
        .filter(|x| *x == 0.0 || x.is_normal() || !x.is_finite())
        .collect()
}

/// Drop inputs below `floor`, keeping everything at or above it (or non-finite).
#[must_use]
pub fn above_flush_floor(inputs: &[f32], floor: f32) -> Vec<f32> {
    inputs
        .iter()
        .copied()
        .filter(|x| !x.is_finite() || *x >= floor)
        .collect()
}

/// Drop inputs above `cap`, keeping everything at or below it (or non-finite).
#[must_use]
pub fn below_cap(inputs: &[f32], cap: f32) -> Vec<f32> {
    inputs
        .iter()
        .copied()
        .filter(|x| !x.is_finite() || *x <= cap)
        .collect()
}

/// Distance of a device `f32` result from its `f64` value-function reference,
/// in ULP of `f32` at the reference's magnitude. Never NaN.
#[must_use]
pub fn ulps_from_reference(got: f32, reference: f64) -> f64 {
    if reference.is_nan() {
        return if got.is_nan() { 0.0 } else { f64::INFINITY };
    }
    if reference.is_infinite() {
        return if f64::from(got) == reference {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if got.is_infinite() {
        // Correct overflow to infinity scores 0: the next float past FLT_MAX
        // is +inf, so a reference past the midpoint overflows by rounding.
        let midpoint = f64::from(f32::MAX) + 2f64.powi(103);
        return if (got.is_sign_positive() && reference > midpoint)
            || (got.is_sign_negative() && reference < -midpoint)
        {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if got.is_nan() {
        // The reference is finite here (NaN and infinite references returned
        // above), so a NaN result is always an infinite miss, never a silent
        // NaN comparison in the runner.
        return f64::INFINITY;
    }
    if reference == 0.0 {
        return f64::from(got.abs()) / 2f64.powi(-149);
    }
    let magnitude = reference.abs();
    let ulp = if magnitude < f64::from(f32::MIN_POSITIVE) {
        2f64.powi(-149)
    } else {
        let biased = i32::try_from((magnitude.to_bits() >> 52) & 0x7ff)
            .expect("invariant: an 11-bit exponent field fits in i32");
        let significand =
            i32::try_from(f32::MANTISSA_DIGITS).expect("invariant: 24 significand bits fit in i32");
        2f64.powi(biased - 1023 - (significand - 1))
    };
    (f64::from(got) - reference).abs() / ulp
}

/// `dense` linspace points over `[lo, hi]` plus explicit `edges` and the
/// `±0`, `±inf`, `NaN` specials every value function defines.
#[must_use]
pub fn sweep_inputs(lo: f32, hi: f32, dense: usize, edges: &[f32]) -> Vec<f32> {
    let mut inputs = Vec::with_capacity(dense + edges.len() + 4);
    for index in 0..dense {
        let t = f32::from_count(index) / f32::from_count(dense - 1);
        inputs.push(lo + (hi - lo) * t);
    }
    inputs.extend_from_slice(edges);
    inputs.extend_from_slice(&[-0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
    inputs
}

/// Sweep `Op` over `inputs` on one backend, asserting every result is within
/// `max_ulp` of the `f64` value-function reference.
pub fn assert_unary_sweep_within<D, E, Op>(
    device: &D,
    ops: &E,
    op_name: &'static str,
    inputs: &[f32],
    max_ulp: f64,
) where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    Op: UnaryExpr<E::Dialect> + UnaryValue,
{
    let name = device.backend_name();
    let input = device.upload(inputs).expect("sweep input upload");
    let output = device
        .alloc_zeroed::<f32>(inputs.len())
        .expect("sweep output alloc");
    let layout = Layout::c_contiguous([inputs.len()]).expect("rank-1 sweep layout");
    ops.unary_into::<Op, 1>(
        device,
        StridedView::new(&input, &layout),
        StridedView::new(&output, &layout),
    )
    .expect("sweep dispatch");
    let mut got = vec![0.0f32; inputs.len()];
    device.download(&output, &mut got).expect("sweep download");
    let mut worst = 0.0f64;
    let mut worst_input = 0.0f32;
    let mut over = 0usize;
    for (&x, &y) in inputs.iter().zip(&got) {
        let error = ulps_from_reference(y, Op::apply(f64::from(x)));
        if error > worst {
            worst = error;
            worst_input = x;
        }
        if error > max_ulp {
            over += 1;
        }
    }
    assert!(
        worst <= max_ulp,
        "{name}: {op_name} sweep worst {worst} ULP at x = {worst_input:e} \
         ({over} of {} over the bound {max_ulp})",
        inputs.len()
    );
}

/// Sweep `Op` over `inputs` on one backend, asserting every absolute error
/// is within `abs_floor` of the `f64` value-function reference.
///
/// The absolute floor covers neighborhoods of value-function roots, where
/// relative ULP scoring is vacuous: near a simple root any fixed absolute
/// form error scores arbitrarily many ULP, so the root window sweeps here
/// while the surrounding bands sweep relatively (see `gelu_grad`). The
/// floor derives from the form's absolute rounding analysis like any
/// bound. Non-finite results against a finite reference fail loudly;
/// NaN-in/NaN-out and exact infinities score zero as in
/// [`ulps_from_reference`].
pub fn assert_unary_sweep_abs_within<D, E, Op>(
    device: &D,
    ops: &E,
    op_name: &'static str,
    inputs: &[f32],
    abs_floor: f64,
) where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    Op: UnaryExpr<E::Dialect> + UnaryValue,
{
    let name = device.backend_name();
    let input = device.upload(inputs).expect("sweep input upload");
    let output = device
        .alloc_zeroed::<f32>(inputs.len())
        .expect("sweep output alloc");
    let layout = Layout::c_contiguous([inputs.len()]).expect("rank-1 sweep layout");
    ops.unary_into::<Op, 1>(
        device,
        StridedView::new(&input, &layout),
        StridedView::new(&output, &layout),
    )
    .expect("sweep dispatch");
    let mut got = vec![0.0f32; inputs.len()];
    device.download(&output, &mut got).expect("sweep download");
    let mut worst = 0.0f64;
    let mut worst_input = 0.0f32;
    let mut over = 0usize;
    for (&x, &y) in inputs.iter().zip(&got) {
        let reference = Op::apply(f64::from(x));
        let error = if reference.is_nan() {
            if y.is_nan() { 0.0 } else { f64::INFINITY }
        } else if reference.is_infinite() {
            if f64::from(y) == reference {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            let abs_err = (f64::from(y) - reference).abs();
            if abs_err.is_nan() {
                f64::INFINITY
            } else {
                abs_err
            }
        };
        if error > worst {
            worst = error;
            worst_input = x;
        }
        if error > abs_floor {
            over += 1;
        }
    }
    assert!(
        worst <= abs_floor,
        "{name}: {op_name} sweep worst {worst:e} absolute at x = {worst_input:e} \
         ({over} of {} over the floor {abs_floor:e})",
        inputs.len()
    );
}

/// Sweep `Op` with fixed `parameters` over `inputs` on one backend,
/// asserting every result is within `max_ulp` of the `f64` value-function
/// reference.
pub fn assert_parameterized_sweep_within<D, P, Op>(
    device: &D,
    ops: &P,
    op_name: &'static str,
    parameters: [f32; 2],
    inputs: &[f32],
    max_ulp: f64,
) where
    D: ComputeDevice,
    P: ParameterizedUnaryOps<D>,
    Op: ParameterizedUnaryExpr<P::Dialect> + ParameterizedUnaryValue,
{
    let name = device.backend_name();
    let input = device.upload(inputs).expect("sweep input upload");
    let output = device
        .alloc_zeroed::<f32>(inputs.len())
        .expect("sweep output alloc");
    let layout = Layout::c_contiguous([inputs.len()]).expect("rank-1 sweep layout");
    ops.parameterized_unary_into::<Op, 1>(
        device,
        StridedView::new(&input, &layout),
        parameters,
        StridedView::new(&output, &layout),
    )
    .expect("sweep dispatch");
    let mut got = vec![0.0f32; inputs.len()];
    device.download(&output, &mut got).expect("sweep download");
    let (first, second) = (f64::from(parameters[0]), f64::from(parameters[1]));
    let mut worst = 0.0f64;
    let mut worst_input = 0.0f32;
    let mut over = 0usize;
    for (&x, &y) in inputs.iter().zip(&got) {
        let error = ulps_from_reference(y, Op::apply(f64::from(x), first, second));
        if error > worst {
            worst = error;
            worst_input = x;
        }
        if error > max_ulp {
            over += 1;
        }
    }
    assert!(
        worst <= max_ulp,
        "{name}: {op_name} sweep worst {worst} ULP at x = {worst_input:e} \
         with parameters {parameters:?} ({over} of {} over the bound {max_ulp})",
        inputs.len()
    );
}
