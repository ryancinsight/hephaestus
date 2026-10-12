use crate::domain::device::ComputeDevice;
use crate::domain::dialect::KernelDialect;
use crate::domain::error::Result;
use crate::domain::ops::{CombineExpr, IdentityToken, MaxOp, MinOp, OpIdentity, ProdOp, SumOp};
use crate::domain::vector::DenseVectorOps;
use crate::domain::view::StridedView;
use eunomia::Pod;

/// Device-neutral rank-2 axis reductions over strided operands.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `R: AxisReductionOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Reduced shape
///
/// Reducing a rank-2 operand along `axis` preserves that axis at length one, so
/// a `[3, 4]` input reduced along axis `0` writes a `[1, 4]` output. Keeping the
/// rank makes the result directly broadcastable against the input, which is why
/// the reduced axis is not dropped.
///
/// # Prepared reductions
///
/// [`Self::prepare_reduce_axis_into`] binds dispatch resources to a fixed
/// operand pair once. The handle observes later writes to the buffers it was
/// prepared over — it holds the allocations, not a snapshot — so an iterative
/// consumer re-dispatches without re-planning and without allocating.
///
/// # Operator genericity
///
/// The combining operator is a type parameter bounded by the implementor's own
/// [`Self::Dialect`], because the shader expression for a reduction is
/// dialect-specific (`Wgsl`, `CudaC`, `HipC`) while the reduction contract is
/// not. That keeps one seam across backends whose kernels are written in
/// different languages.
///
/// # Special values
///
/// Floating-point NaN and infinity behaviour follows the kernel
/// dialect's declared capability: see
/// [`KernelDialect::IEEE_SPECIAL_VALUES`]
/// (ADR 0043) for what is and is not promised per dialect.
pub trait AxisReductionOps<D: ComputeDevice, T: Pod> {
    /// Kernel dialect this backend authors reductions in.
    type Dialect: KernelDialect;

    /// Dispatch resources bound to one input/output operand pair.
    ///
    /// Lifetime-parameterized so backends whose prepared form borrows the
    /// operand pair (CUDA, ROCm) implement the seam without erasing that
    /// borrow; handle-based backends (WGPU) simply ignore `'op`.
    type Prepared<'op>
    where
        Self: 'op,
        D: 'op,
        T: 'op;

    /// Reduce `input` along `axis` into `output` under the combining operator
    /// `Op`.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when the axis is out of range, the output
    /// shape does not match the reduced shape, a layout is unsupported, or the
    /// backend dispatch fails.
    fn reduce_axis_into<Op>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, D::Buffer<T>, 2>,
    ) -> Result<()>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>,
    {
        let prepared = self.prepare_reduce_axis_into::<Op>(device, input, axis, output)?;
        self.dispatch_prepared(device, &prepared)
    }

    /// Product-reduce `input` along `axis` into `output`.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when the axis is out of range, the output
    /// shape does not match the reduced shape, a layout is unsupported, or the
    /// backend dispatch fails.
    fn prod_axis_into(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, D::Buffer<T>, 2>,
    ) -> Result<()>
    where
        ProdOp: CombineExpr<Self::Dialect>,
        T: OpIdentity<ProdOp> + IdentityToken<ProdOp, Self::Dialect>,
    {
        let prepared = self.prepare_reduce_axis_into::<ProdOp>(device, input, axis, output)?;
        self.dispatch_prepared(device, &prepared)
    }

    /// Mean-reduce `input` along `axis` into `output`.
    ///
    /// The divisor is the reduced axis length, so the reduced axis must be
    /// non-empty.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when the axis is out of range or empty,
    /// the output shape does not match the reduced shape, a layout is
    /// unsupported, or the backend dispatch fails.
    fn mean_axis_into(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, D::Buffer<T>, 2>,
    ) -> Result<()>
    where
        SumOp: CombineExpr<Self::Dialect>,
        T: OpIdentity<SumOp> + IdentityToken<SumOp, Self::Dialect>;

    /// Min-reduce `input` along `axis` into `output`.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when the axis is out of range, the
    /// output shape does not match the reduced shape, a layout is unsupported,
    /// or the backend dispatch fails.
    fn min_axis_into(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, D::Buffer<T>, 2>,
    ) -> Result<()>
    where
        MinOp: CombineExpr<Self::Dialect>,
        T: OpIdentity<MinOp> + IdentityToken<MinOp, Self::Dialect>,
    {
        let prepared = self.prepare_reduce_axis_into::<MinOp>(device, input, axis, output)?;
        self.dispatch_prepared(device, &prepared)
    }

    /// Max-reduce `input` along `axis` into `output`.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when the axis is out of range, the
    /// output shape does not match the reduced shape, a layout is unsupported,
    /// or the backend dispatch fails.
    fn max_axis_into(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, D::Buffer<T>, 2>,
    ) -> Result<()>
    where
        MaxOp: CombineExpr<Self::Dialect>,
        T: OpIdentity<MaxOp> + IdentityToken<MaxOp, Self::Dialect>,
    {
        let prepared = self.prepare_reduce_axis_into::<MaxOp>(device, input, axis, output)?;
        self.dispatch_prepared(device, &prepared)
    }

    /// Bind dispatch resources for reducing `input` along `axis` into `output`
    /// under the combining operator `Op`.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when axis, shape, output-layout, or
    /// aliasing validation fails.
    fn prepare_reduce_axis_into<'op, Op>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<T>, 2>,
        axis: usize,
        output: StridedView<'op, D::Buffer<T>, 2>,
    ) -> Result<Self::Prepared<'op>>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>;

    /// Re-dispatch a prepared reduction over its bound operands.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch failure.
    fn dispatch_prepared(&self, device: &D, prepared: &Self::Prepared<'_>) -> Result<()>;
}

/// Device-neutral full-array reduction over a strided n-D operand to a
/// single scalar stored in a one-element output buffer.
///
/// This complements [`crate::AxisReductionOps`] by reducing the entire operand to
/// one value rather than along one axis. The output is a rank-1 view of length
/// one, which keeps the same `StridedView` output shape and lets consumers
/// place the scalar where they need it.
///
/// Implementors are zero-sized per-backend markers. A bound of
/// `R: FullReductionOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Special values
///
/// Floating-point NaN and infinity behaviour follows the kernel
/// dialect's declared capability: see
/// [`KernelDialect::IEEE_SPECIAL_VALUES`]
/// (ADR 0043) for what is and is not promised per dialect.
pub trait FullReductionOps<D: ComputeDevice, T: Pod> {
    /// Kernel dialect this backend authors reductions in.
    type Dialect: KernelDialect;

    /// Prepared resources for a full reduction bound to fixed input/output
    /// views.
    ///
    /// Lifetime-parameterized so backends whose prepared form borrows the
    /// operand pair implement the seam without erasing that borrow;
    /// handle-based backends simply ignore `'op`.
    type Prepared<'op, const N: usize>
    where
        Self: 'op,
        D: 'op,
        T: 'op;

    /// Reduce the entire `input` view into the one-element `output` under the
    /// combining operator `Op`.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch when `output` does not have length one, an
    /// aliasing violation, a layout validation failure, or the backend dispatch
    /// failure.
    fn reduce_full_into<Op, const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        output: StridedView<'_, D::Buffer<T>, 1>,
    ) -> Result<()>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>,
    {
        let prepared = self.prepare_reduce_full::<Op, N>(device, input, output)?;
        self.dispatch_full::<N>(device, &prepared)
    }

    /// Bind dispatch resources for reducing the entire `input` view into the
    /// one-element `output` under the combining operator `Op`.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch when `output` does not have length one, an
    /// aliasing violation, a layout validation failure, or the backend
    /// preparation failure.
    fn prepare_reduce_full<'op, Op, const N: usize>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<T>, N>,
        output: StridedView<'op, D::Buffer<T>, 1>,
    ) -> Result<Self::Prepared<'op, N>>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>;

    /// Re-dispatch a prepared full reduction over its bound operands.
    ///
    /// # Errors
    ///
    /// Returns a prepared-operand mismatch or the backend dispatch failure.
    fn dispatch_full<const N: usize>(
        &self,
        device: &D,
        prepared: &Self::Prepared<'_, N>,
    ) -> Result<()>;
}

/// Marker asserting that a backend supplies both halves of the accelerator
/// operand surface a solver needs: dense vector recurrences and axis
/// reductions over strided views.
///
/// Consumers bind to this instead of restating both bounds, and a backend that
/// implements both acquires it through the blanket implementation below.
pub trait StridedComputeBackend<D: ComputeDevice, T: Pod>:
    DenseVectorOps<D, T> + AxisReductionOps<D, T>
{
}

impl<B, D, T> StridedComputeBackend<D, T> for B
where
    D: ComputeDevice,
    T: Pod,
    B: DenseVectorOps<D, T> + AxisReductionOps<D, T>,
{
}
