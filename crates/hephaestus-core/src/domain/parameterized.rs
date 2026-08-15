//! Runtime-parameter unary expressions and their device-neutral dispatch seam.
//!
//! Parameter values remain dispatch data. They never enter generated source or
//! pipeline-cache keys, so changing activation bounds reuses the compiled
//! kernel and changes only the two scalar arguments.

use leto::Layout;

use super::device::ComputeDevice;
use super::dialect::{CudaC, HipC, KernelDialect, Wgsl};
use super::error::{HephaestusError, Result};
use super::view::StridedView;

/// Unary expression over `x` and two runtime scalars named `first` and
/// `second` in dialect `L`.
pub trait ParameterizedUnaryExpr<L: KernelDialect>: Copy + Send + Sync + 'static {
    /// Expression mapping `x`, `first`, and `second` to one output value.
    const EXPR: &'static str;
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
    };
}

impl_c_family!(CudaC);
impl_c_family!(HipC);

#[cfg(test)]
mod tests {
    use super::*;

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
