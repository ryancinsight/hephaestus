//! Device-neutral runtime-rank expression-fusion contracts.
//!
//! Static operation markers remain the right representation for a fixed
//! operation family. Consumer expression graphs are different: their source,
//! rank, and input count are runtime values. This module carries only that
//! variation across the provider boundary; a backend still monomorphizes the
//! dispatch contract over its device, scalar, dialect, and expression adapter.

use std::borrow::Cow;

use bytemuck::Pod;

use super::{
    device::ComputeDevice, dialect::KernelDialect, error::Result, view::DynamicStridedView,
};

/// A consumer-owned expression fragment for a kernel dialect.
///
/// The returned source is one expression, not a statement or a complete
/// shader. Providers bind the canonical locals `input_0`, `input_1`, and so on
/// before interpolating it into their owned kernel wrapper. Returning
/// [`Cow`] lets static expressions avoid allocation while dynamic expression
/// graphs hand over an owned string without a second copy.
pub trait FusedExpression<L: KernelDialect>: Send + Sync {
    /// Return the dialect-specific expression fragment.
    fn source(&self) -> Cow<'_, str>;
}

/// Closed reduction vocabulary for fused expression evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FusedReduction {
    /// Add all values along the selected axis.
    Sum,
    /// Multiply all values along the selected axis.
    Product,
    /// Average all values along the selected axis.
    Mean,
    /// Select the greatest value along the selected axis.
    Maximum,
    /// Select the least value along the selected axis.
    Minimum,
}

/// Device-neutral one-shot elementwise fusion over runtime-rank views.
///
/// Providers own expression lowering, layout metadata, and execution. The
/// borrowed input slice is the only variable-size part of this contract; no
/// tensor, layout, or device handle is copied by the seam itself.
pub trait FusedElementwiseOps<D: ComputeDevice, T: Pod> {
    /// Kernel dialect authored by the provider.
    type Dialect: KernelDialect;

    /// Evaluate `expression` over the logical output view.
    ///
    /// Each input must broadcast to `output.layout`; the provider validates
    /// that rule and the device-buffer ownership before dispatch.
    ///
    /// # Errors
    ///
    /// Returns a typed provider error when the expression, views, device
    /// resources, or dispatch plan is invalid.
    fn fused_elementwise_into<E>(
        &self,
        device: &D,
        expression: &E,
        inputs: &[DynamicStridedView<'_, D::Buffer<T>>],
        output: DynamicStridedView<'_, D::Buffer<T>>,
    ) -> Result<()>
    where
        E: FusedExpression<Self::Dialect>;
}

/// Device-neutral one-shot fused expression reduction over a runtime-rank
/// output view.
pub trait FusedReductionOps<D: ComputeDevice, T: Pod> {
    /// Kernel dialect authored by the provider.
    type Dialect: KernelDialect;

    /// Evaluate `expression` and reduce its logical result along `axis`.
    ///
    /// The output keeps the reduced axis at extent one. Providers validate the
    /// broadcasted expression shape, axis, output layout, and reduction's
    /// empty-axis semantics before dispatch.
    ///
    /// # Errors
    ///
    /// Returns a typed provider error when the expression, views, reduction,
    /// device resources, or dispatch plan is invalid.
    fn fused_reduce_into<E>(
        &self,
        device: &D,
        expression: &E,
        inputs: &[DynamicStridedView<'_, D::Buffer<T>>],
        reduction: FusedReduction,
        axis: usize,
        output: DynamicStridedView<'_, D::Buffer<T>>,
    ) -> Result<()>
    where
        E: FusedExpression<Self::Dialect>;
}
