//! Sweep runners and input builders shared by the per-operator accuracy modules.

use eunomia::FloatElement;
use hephaestus_core::{
    ComputeDevice, CudaC, DialectScalar, ElementwiseOps, ErfcOp, HipC, ParameterizedUnaryExpr,
    ParameterizedUnaryOps, ParameterizedUnaryValue, StridedView, UnaryExpr, UnaryValue, Wgsl,
};
use leto::Layout;

/// Which published bound family a sweep asserts.
///
/// The backend test passes the variant matching the dialect its seam authors:
/// CUDA tests pass [`SweepDialect::CudaC`], ROCm tests [`SweepDialect::HipC`],
/// and WGSL-authoring backends (WGPU, Metal) [`SweepDialect::Wgsl`]. The
/// runners check the choice against the seam's dialect in debug builds.
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

/// Drop subnormal inputs, keeping `±0`, normals, infinities, and NaN.
///
/// WGSL permits flushing subnormals to zero, and the measured Vulkan driver
/// does (a `1e-40` input reaches the lane as zero, and a subnormal kink
/// comparison flips to the pass-through arm): subnormal inputs are out of
/// contract on WGSL, so the WGSL entries sweep without them. CUDA and HIP
/// behave as IEEE here and keep them.
#[must_use]
pub fn without_subnormals(inputs: &[f32]) -> Vec<f32> {
    inputs
        .iter()
        .copied()
        .filter(|x| *x == 0.0 || x.is_normal() || !x.is_finite())
        .collect()
}

/// Keep inputs whose reference erfc result is normal at the rounded shader
/// argument. This is a coverage criterion, not a portable error-bound proof.
/// Nonfinite structural controls always remain selected.
pub(super) fn with_normal_erfc_result(inputs: &[f32], argument: impl Fn(f32) -> f32) -> Vec<f32> {
    inputs
        .iter()
        .copied()
        .filter(|&x| {
            !x.is_finite() || ErfcOp::apply(f64::from(argument(x))) >= f64::from(f32::MIN_POSITIVE)
        })
        .collect()
}

/// Keep finite inputs with zero or normal final reference results, plus the
/// nonfinite structural controls. Intermediate normality must not exclude
/// normal final results from the WGSL conformance sweep.
pub(super) fn with_normal_unary_result<Op: UnaryValue>(inputs: &[f32]) -> Vec<f32> {
    inputs
        .iter()
        .copied()
        .filter(|&x| {
            let reference = Op::apply(f64::from(x));
            !x.is_finite() || reference == 0.0 || reference.abs() >= f64::from(f32::MIN_POSITIVE)
        })
        .collect()
}

/// Drop inputs below `floor`, keeping everything at or above it.
///
/// WGSL permits flushing subnormal results to zero: ops whose true results
/// go subnormal on the negative tail (softplus and the sigmoid below
/// `x = -87`, where the result crosses `f32::MIN_POSITIVE`) sweep WGSL
/// without that tail, since a flushed zero scores tens of thousands of
/// subnormal ULP against a subnormal reference through no fault of the
/// form. CUDA and HIP behave as IEEE here and keep the tail.
#[must_use]
pub fn above_flush_floor(inputs: &[f32], floor: f32) -> Vec<f32> {
    // Non-finite inputs always pass through: the `±inf`/`NaN` specials
    // verify the guards, and the floor only concerns finite inputs whose
    // results would flush.
    inputs
        .iter()
        .copied()
        .filter(|x| !x.is_finite() || *x >= floor)
        .collect()
}

/// Drop inputs above `cap`, keeping everything at or below it.
///
/// WGSL permits flushing subnormal results to zero: ops whose true results
/// go subnormal on a positive tail sweep WGSL
/// without that tail, since a flushed zero scores thousands of subnormal
/// ULP against a subnormal reference through no fault of the form. CUDA
/// and HIP behave as IEEE here and keep the tail.
#[must_use]
pub fn below_cap(inputs: &[f32], cap: f32) -> Vec<f32> {
    // Non-finite inputs always pass through: the `±inf`/`NaN` specials
    // verify the guards, and the cap only concerns finite inputs whose
    // results would flush.
    inputs
        .iter()
        .copied()
        .filter(|x| !x.is_finite() || *x <= cap)
        .collect()
}

/// Distance of a device `f32` result from its `f64` value-function reference,
/// in ULP of `f32` at the reference's magnitude. Never NaN: a NaN `got`
/// against a finite reference scores infinity.
///
/// - NaN reference: `0` when `got` is NaN, infinity otherwise.
/// - Infinite reference: `0` when `got` equals it, infinity otherwise.
/// - Infinite `got` against a finite reference: `0` for a correct overflow
///   to infinity, infinity otherwise.
/// - Zero reference: absolute error in units of the subnormal ULP `2^-149`,
///   so a form returning `±0` scores `0` and anything else scores enormously.
/// - Otherwise: `|got − ref| / ulp32(ref)`, with `|ref|` below normal clamped
///   to the `2^-149` floor (the sweeps pin tails whose references are normal
///   floats, where the scale is exact).
#[must_use]
pub fn ulps_from_f64(got: f32, reference: f64) -> f64 {
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
        let error = ulps_from_f64(y, Op::apply(f64::from(x)));
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
/// [`ulps_from_f64`].
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
        let error = ulps_from_f64(y, Op::apply(f64::from(x), first, second));
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
