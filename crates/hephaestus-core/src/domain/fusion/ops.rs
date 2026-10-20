//! Device-neutral one-shot fusion seams over runtime-rank views.

use std::borrow::Cow;
use std::sync::Arc;

use eunomia::Pod;

use crate::{ComputeDevice, DynamicStridedView, KernelDialect, Result};

/// Maximum runtime rank represented by the provider fusion ABI.
///
/// This is a provider representation limit shared by every current backend;
/// Leto itself remains arbitrary-rank.
pub const MAX_FUSION_RANK: usize = 8;
/// Threads per one-dimensional fusion launch block.
pub const FUSION_WORKGROUP_WIDTH: u32 = 256;

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

/// Collision-safe identity for a runtime-generated fusion kernel.
///
/// The complete expression is retained in the key. A truncated digest would
/// let two distinct expressions reuse the wrong compiled kernel.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FusionPipelineKey {
    /// Kernel family marker: elementwise and reduction kernels share the
    /// expression namespace but never the compiled artifact.
    pub family: std::any::TypeId,
    /// Host scalar the kernel was generated for.
    pub scalar: std::any::TypeId,
    /// Runtime rank the generated indexing code covers.
    pub rank: u32,
    /// Number of tensor inputs bound by the generated kernel.
    pub input_count: u32,
    /// Reduction the kernel applies, or [`None`] for elementwise kernels.
    pub reduction: Option<FusedReduction>,
    /// Complete validated expression source.
    pub expression: Arc<str>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_key_compares_complete_expression_source() {
        let base = FusionPipelineKey {
            family: std::any::TypeId::of::<()>(),
            scalar: std::any::TypeId::of::<f32>(),
            rank: 2,
            input_count: 2,
            reduction: None,
            expression: Arc::from("input_0 + input_1"),
        };
        let changed = FusionPipelineKey {
            expression: Arc::from("input_0 - input_1"),
            ..base.clone()
        };
        assert_ne!(base, changed);
        assert_eq!(base, base.clone());
    }
}
