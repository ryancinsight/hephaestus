//! The device-neutral dispatch seam for runtime-parameter unary operations.

use super::ParameterizedUnaryExpr;
use crate::domain::device::ComputeDevice;
use crate::domain::dialect::KernelDialect;
use crate::domain::error::Result;
use crate::domain::view::StridedView;

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
