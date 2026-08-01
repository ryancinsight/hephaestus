//! Backend-neutral elementwise operations over strided n-D views.
//!
//! Every backend crate already provides unary and binary elementwise kernels,
//! but as free functions over its own concrete device and operand types. A
//! consumer generic over the device could reach no elementwise operation
//! without binding to one backend. This trait closes that gap: a tensor
//! consumer programs against `E: ElementwiseOps<D, T>` and runs on every
//! backend implementing it.
//!
//! # Shape
//!
//! `Self` is a backend-provided bundle of prepared kernels, constructed once
//! against a device, and the device is passed per call. All operands are
//! `StridedView`s over device buffers, so layouts with arbitrary strides and
//! offsets are supported wherever the backend's kernel allows.
//!
//! # Contract
//!
//! Unary operations read every element of the input view and write the
//! transformed value into the output view. Binary operations read aligned
//! elements of both input views and write the combined value into the output
//! view. Output views must not alias input views; implementations reject
//! aliasing rather than producing undefined results. Empty views are a no-op.
//!
//! # Operator genericity
//!
//! The elementwise expression is a type parameter bounded by the implementor's
//! own [`crate::ElementwiseOps`] `Dialect`, because the shader expression is
//! dialect-specific while the elementwise contract is not. That keeps one seam
//! across backends whose kernels are written in different languages.

use bytemuck::Pod;

use super::device::ComputeDevice;
use super::dialect::{DialectScalar, KernelDialect};
use super::error::Result;
use super::ops::{BinaryExpr, TypedBinaryExpr, UnaryExpr};
use super::view::StridedView;

/// Device-neutral elementwise operations over strided n-D views.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `E: ElementwiseOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Special values
///
/// Floating-point NaN and infinity behaviour follows the kernel
/// dialect's declared capability: see
/// [`KernelDialect::IEEE_SPECIAL_VALUES`](crate::KernelDialect::IEEE_SPECIAL_VALUES)
/// (ADR 0043) for what is and is not promised per dialect.
pub trait ElementwiseOps<D: ComputeDevice, T: Pod> {
    /// Kernel dialect this backend authors elementwise kernels in.
    type Dialect: KernelDialect;

    /// Prepared resources for a unary elementwise operation bound to fixed
    /// input and output views.
    type PreparedUnary<'op, const N: usize>
    where
        Self: 'op,
        D: 'op,
        T: 'op;

    /// Prepared resources for a binary elementwise operation bound to fixed
    /// left, right, and output views.
    type PreparedBinary<'op, const N: usize>
    where
        Self: 'op,
        D: 'op,
        T: 'op;

    /// Prepared resources for a broadcast-scalar binary elementwise
    /// operation bound to fixed input/output views. The scalar itself is
    /// dispatch data captured at preparation.
    type PreparedScalar<'op, const N: usize>
    where
        Self: 'op,
        D: 'op,
        T: 'op;

    /// Prepared resources for a scalar-aware binary elementwise operation bound
    /// to fixed left, right, and output views.
    type PreparedTypedBinary<'op, const N: usize>
    where
        Self: 'op,
        D: 'op,
        T: 'op;

    /// Compute `output = Op(input)` elementwise in one shot.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch, an aliased output, a layout validation
    /// failure, or the backend dispatch failure.
    fn unary_into<Op, const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>
    where
        Op: UnaryExpr<Self::Dialect>,
    {
        let prepared = self.prepare_unary_into::<Op, N>(device, input, output)?;
        self.dispatch_unary::<N>(device, &prepared)
    }

    /// Prepare a unary elementwise dispatch bound to `input` and `output`.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch, an aliased output, a layout validation
    /// failure, or the backend preparation failure.
    fn prepare_unary_into<'op, Op, const N: usize>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<T>, N>,
        output: StridedView<'op, D::Buffer<T>, N>,
    ) -> Result<Self::PreparedUnary<'op, N>>
    where
        Op: UnaryExpr<Self::Dialect>;

    /// Re-dispatch a prepared unary operation over its bound operands.
    ///
    /// # Errors
    ///
    /// Returns a prepared-operand mismatch or the backend dispatch failure.
    fn dispatch_unary<const N: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedUnary<'_, N>,
    ) -> Result<()>;

    /// Compute `output = Op(lhs, rhs)` elementwise in one shot.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch, an aliased output, a layout validation
    /// failure, or the backend dispatch failure.
    fn binary_into<Op, const N: usize>(
        &self,
        device: &D,
        lhs: StridedView<'_, D::Buffer<T>, N>,
        rhs: StridedView<'_, D::Buffer<T>, N>,
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        let prepared = self.prepare_binary_into::<Op, N>(device, lhs, rhs, output)?;
        self.dispatch_binary::<N>(device, &prepared)
    }

    /// Prepare a binary elementwise dispatch bound to `lhs`, `rhs`, and
    /// `output`.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch, an aliased output, a layout validation
    /// failure, or the backend preparation failure.
    fn prepare_binary_into<'op, Op, const N: usize>(
        &self,
        device: &D,
        lhs: StridedView<'op, D::Buffer<T>, N>,
        rhs: StridedView<'op, D::Buffer<T>, N>,
        output: StridedView<'op, D::Buffer<T>, N>,
    ) -> Result<Self::PreparedBinary<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>;

    /// Re-dispatch a prepared binary operation over its bound operands.
    ///
    /// # Errors
    ///
    /// Returns a prepared-operand mismatch or the backend dispatch failure.
    fn dispatch_binary<const N: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedBinary<'_, N>,
    ) -> Result<()>;

    /// Compute `output = Op(input, scalar)` elementwise, broadcasting one
    /// runtime scalar as the right-hand operand.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch, an aliased output, a layout validation
    /// failure, or the backend dispatch failure.
    fn scalar_into<Op, const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        scalar: T,
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        let prepared = self.prepare_scalar_into::<Op, N>(device, input, scalar, output)?;
        self.dispatch_scalar::<N>(device, &prepared)
    }

    /// Prepare a broadcast-scalar dispatch bound to `input` and `output`.
    /// The scalar value is captured by the prepared form; rebind semantics
    /// apply to the buffer operands only.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch, an aliased output, a layout validation
    /// failure, or the backend preparation failure.
    fn prepare_scalar_into<'op, Op, const N: usize>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<T>, N>,
        scalar: T,
        output: StridedView<'op, D::Buffer<T>, N>,
    ) -> Result<Self::PreparedScalar<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>;

    /// Re-dispatch a prepared broadcast-scalar operation over its bound
    /// operands.
    ///
    /// # Errors
    ///
    /// Returns a prepared-operand mismatch or the backend dispatch failure.
    fn dispatch_scalar<const N: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedScalar<'_, N>,
    ) -> Result<()>;

    /// Compute `output = Op(lhs, rhs)` elementwise in one shot using a
    /// scalar-aware expression (e.g. comparisons with dialect-specific mask
    /// literals).
    ///
    /// # Type parameters
    ///
    /// `T` must implement [`DialectScalar`] for [`Self::Dialect`] so the
    /// operation can emit the correct literal tokens for the scalar type.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch, an aliased output, a layout validation
    /// failure, or the backend dispatch failure.
    fn typed_binary_into<Op, const N: usize>(
        &self,
        device: &D,
        lhs: StridedView<'_, D::Buffer<T>, N>,
        rhs: StridedView<'_, D::Buffer<T>, N>,
        output: StridedView<'_, D::Buffer<T>, N>,
    ) -> Result<()>
    where
        Op: TypedBinaryExpr<Self::Dialect, T>,
        T: DialectScalar<Self::Dialect>,
    {
        let prepared = self.prepare_typed_binary_into::<Op, N>(device, lhs, rhs, output)?;
        self.dispatch_typed_binary::<N>(device, &prepared)
    }

    /// Prepare a scalar-aware binary elementwise dispatch bound to `lhs`,
    /// `rhs`, and `output`.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch, an aliased output, a layout validation
    /// failure, or the backend preparation failure.
    fn prepare_typed_binary_into<'op, Op, const N: usize>(
        &self,
        device: &D,
        lhs: StridedView<'op, D::Buffer<T>, N>,
        rhs: StridedView<'op, D::Buffer<T>, N>,
        output: StridedView<'op, D::Buffer<T>, N>,
    ) -> Result<Self::PreparedTypedBinary<'op, N>>
    where
        Op: TypedBinaryExpr<Self::Dialect, T>,
        T: DialectScalar<Self::Dialect>;

    /// Re-dispatch a prepared typed binary operation over its bound operands.
    ///
    /// # Errors
    ///
    /// Returns a prepared-operand mismatch or the backend dispatch failure.
    fn dispatch_typed_binary<const N: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedTypedBinary<'_, N>,
    ) -> Result<()>;
}
