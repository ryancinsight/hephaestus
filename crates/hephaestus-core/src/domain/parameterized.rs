//! Runtime-parameter unary expressions and their device-neutral dispatch seam.
//!
//! Parameter values remain dispatch data. They never enter generated source or
//! pipeline-cache keys, so changing activation bounds reuses the compiled
//! kernel and changes only the two scalar arguments.

use bytemuck::Pod;
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

/// Device-neutral runtime-parameter unary operations over strided views.
///
/// The pair is operation-defined. Hardtanh interprets it as `(minimum,
/// maximum)` and Threshold as `(threshold, replacement)`. Gradient operations
/// ignore the unused second value while retaining one stable dispatch shape.
pub trait ParameterizedUnaryOps<D: ComputeDevice, T: Pod> {
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
        input: StridedView<'_, D::Buffer<T>, N>,
        parameters: [T; 2],
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>
    where
        Op: ParameterizedUnaryExpr<Self::Dialect>;
}

/// Validate that a writable parameterized-unary layout is addressable and
/// proves non-overlap, then return its logical element count.
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

    let mut axes = [(0_usize, 0_usize); N];
    let mut active = 0;
    for (&extent, &stride) in layout.shape.iter().zip(&layout.strides) {
        if extent <= 1 {
            continue;
        }
        let magnitude = stride.unsigned_abs();
        if magnitude == 0 {
            return Err(nonoverlap_error());
        }
        axes[active] = (magnitude, extent);
        active += 1;
    }
    axes[..active].sort_unstable_by_key(|&(stride, _)| stride);

    let mut covered_span = 1_usize;
    for &(stride, extent) in &axes[..active] {
        if stride < covered_span {
            return Err(nonoverlap_error());
        }
        covered_span = (extent - 1)
            .checked_mul(stride)
            .and_then(|axis_span| covered_span.checked_add(axis_span))
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: "writable output layout span overflows".to_string(),
            })?;
    }
    Ok(len)
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
}
