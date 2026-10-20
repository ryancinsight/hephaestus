//! The WGPU dense vector-operation seam.
//!
//! The prepared handles, their inherent helpers, and both seam impls share
//! this module because they share private state: splitting them would only
//! widen field and helper visibility across the boundary.

use core::marker::PhantomData;

use eunomia::Pod;
use hephaestus_core::RetainedReductions;
use hephaestus_core::{
    Binding, CommandStream, ComputeDevice, DenseVectorOps, DeviceBuffer, DispatchGrid,
    HephaestusError, IdentityToken, KernelDevice, MaxOp, OpIdentity, Result, VectorFactor,
    VectorScalar, Wgsl,
};

use hephaestus_core::{AddOp, BlockWidth, DivOp, MulOp, SubOp};
use leto::Layout;

use crate::{
    L2NormScalar, PreparedDot, PreparedL2Norm, StridedOperand, WgpuBuffer, WgpuDevice,
    WgpuPrepared, binary_elementwise_into, prepare_dot, prepare_norm_l2,
};

use super::kernels::{AxpyKernel, ScaleKernel, VectorParams, XpayKernel, workgroup_count};

/// Prepared in-place vector kernels for one WGPU device.
///
/// Shader compilation happens once here rather than per operation, which is
/// what makes the seam usable inside an iteration loop.
pub struct WgpuVectorOps<T> {
    scale: WgpuPrepared<ScaleKernel<T>>,
    axpy: WgpuPrepared<AxpyKernel<T>>,
    xpay: WgpuPrepared<XpayKernel<T>>,
}

/// Prepared dot product bound to the allocations it was built against.
pub struct WgpuPreparedDot<T> {
    operation: PreparedDot<T>,
    left: WgpuBuffer<T>,
    right: WgpuBuffer<T>,
}

/// Prepared Euclidean norm bound to the allocation it was built against.
pub struct WgpuPreparedNorm<T> {
    operation: PreparedL2Norm<T>,
    input: WgpuBuffer<T>,
}

impl<T: VectorScalar<Wgsl> + VectorFactor<Wgsl>> WgpuVectorOps<T> {
    /// Compile the in-place vector kernels on `device`.
    ///
    /// # Errors
    ///
    /// Returns a shader preparation or device failure.
    pub fn new(device: &WgpuDevice) -> Result<Self> {
        Ok(Self {
            scale: device.prepare(&ScaleKernel(PhantomData))?,
            axpy: device.prepare(&AxpyKernel(PhantomData))?,
            xpay: device.prepare(&XpayKernel(PhantomData))?,
        })
    }

    fn dispatch_binary(
        &self,
        device: &WgpuDevice,
        prepared: &WgpuPrepared<AxpyKernel<T>>,
        target: &WgpuBuffer<T>,
        source: &WgpuBuffer<T>,
        factor: T,
    ) -> Result<()> {
        device.dispatch(
            prepared,
            &[Binding::read_write(target), Binding::read(source)],
            &VectorParams::new(factor, target.len())?,
            Self::grid(target.len())?,
        )
    }
}

impl<T> WgpuVectorOps<T> {
    fn require_equal_lengths(left: usize, right: usize) -> Result<()> {
        if left == right {
            Ok(())
        } else {
            Err(HephaestusError::LengthMismatch {
                host_len: left,
                device_len: right,
            })
        }
    }

    /// Reject a prepared handle whose operand is a different allocation than
    /// the one it was bound to, which would otherwise reduce the wrong memory.
    fn require_same_allocation<U: Pod>(
        role: &str,
        expected: &WgpuBuffer<U>,
        actual: &WgpuBuffer<U>,
    ) -> Result<()> {
        if expected.raw() == actual.raw() {
            Ok(())
        } else {
            Err(HephaestusError::DispatchFailed {
                message: format!("prepared {role} received a different device allocation"),
            })
        }
    }

    fn grid(len: usize) -> Result<DispatchGrid> {
        Ok(DispatchGrid::new(workgroup_count(len)?, 1, 1))
    }

    /// Rank-one contiguous layout describing a dense vector operand.
    fn layout(len: usize) -> Result<Layout<1>> {
        Layout::c_contiguous([len]).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("dense vector layout failed: {error}"),
        })
    }

    fn download_scalar<U: L2NormScalar>(device: &WgpuDevice, scalar: &WgpuBuffer<U>) -> Result<U> {
        let mut host = [U::zeroed(); 1];
        device.download(scalar, &mut host)?;
        Ok(host[0])
    }
}

impl<T> DenseVectorOps<WgpuDevice, T> for WgpuVectorOps<T>
where
    T: VectorScalar<Wgsl>
        + VectorFactor<Wgsl>
        + L2NormScalar
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, Wgsl>,
{
    type PreparedDot<'a>
        = WgpuPreparedDot<T>
    where
        Self: 'a;
    type PreparedNorm<'a>
        = WgpuPreparedNorm<T>
    where
        Self: 'a;

    fn copy_vector(
        &self,
        device: &WgpuDevice,
        source: &WgpuBuffer<T>,
        target: &WgpuBuffer<T>,
    ) -> Result<()> {
        Self::require_equal_lengths(source.len(), target.len())?;
        if source.is_empty() {
            return Ok(());
        }
        let mut stream = device.stream()?;
        stream.copy(source, target)?;
        stream.submit()
    }

    fn scale_vector(&self, device: &WgpuDevice, target: &WgpuBuffer<T>, factor: T) -> Result<()> {
        if target.is_empty() {
            return Ok(());
        }
        device.dispatch(
            &self.scale,
            &[Binding::read_write(target)],
            &VectorParams::new(factor, target.len())?,
            Self::grid(target.len())?,
        )
    }

    fn axpy(
        &self,
        device: &WgpuDevice,
        target: &WgpuBuffer<T>,
        source: &WgpuBuffer<T>,
        factor: T,
    ) -> Result<()> {
        Self::require_equal_lengths(target.len(), source.len())?;
        if target.is_empty() {
            return Ok(());
        }
        self.dispatch_binary(device, &self.axpy, target, source, factor)
    }

    fn xpay(
        &self,
        device: &WgpuDevice,
        target: &WgpuBuffer<T>,
        source: &WgpuBuffer<T>,
        factor: T,
    ) -> Result<()> {
        Self::require_equal_lengths(target.len(), source.len())?;
        if target.is_empty() {
            return Ok(());
        }
        device.dispatch(
            &self.xpay,
            &[Binding::read_write(target), Binding::read(source)],
            &VectorParams::new(factor, target.len())?,
            Self::grid(target.len())?,
        )
    }

    fn subtract_into(
        &self,
        device: &WgpuDevice,
        left: &WgpuBuffer<T>,
        right: &WgpuBuffer<T>,
        output: &WgpuBuffer<T>,
    ) -> Result<()> {
        Self::require_equal_lengths(left.len(), right.len())?;
        Self::require_equal_lengths(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<SubOp, T>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn add_into(
        &self,
        device: &WgpuDevice,
        left: &WgpuBuffer<T>,
        right: &WgpuBuffer<T>,
        output: &WgpuBuffer<T>,
    ) -> Result<()> {
        Self::require_equal_lengths(left.len(), right.len())?;
        Self::require_equal_lengths(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<AddOp, T>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn multiply_into(
        &self,
        device: &WgpuDevice,
        left: &WgpuBuffer<T>,
        right: &WgpuBuffer<T>,
        output: &WgpuBuffer<T>,
    ) -> Result<()> {
        Self::require_equal_lengths(left.len(), right.len())?;
        Self::require_equal_lengths(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<MulOp, T>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn divide_into(
        &self,
        device: &WgpuDevice,
        left: &WgpuBuffer<T>,
        right: &WgpuBuffer<T>,
        output: &WgpuBuffer<T>,
    ) -> Result<()> {
        Self::require_equal_lengths(left.len(), right.len())?;
        Self::require_equal_lengths(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<DivOp, T>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn prepare_dot<'a>(
        &self,
        device: &WgpuDevice,
        left: &'a WgpuBuffer<T>,
        right: &'a WgpuBuffer<T>,
    ) -> Result<Self::PreparedDot<'a>> {
        Self::require_equal_lengths(left.len(), right.len())?;
        let layout = Self::layout(left.len())?;
        Ok(WgpuPreparedDot {
            operation: prepare_dot(
                device,
                StridedOperand {
                    buffer: left,
                    layout: &layout,
                },
                StridedOperand {
                    buffer: right,
                    layout: &layout,
                },
            )?,
            left: left.clone(),
            right: right.clone(),
        })
    }

    fn dot_prepared<'a>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedDot<'a>,
        left: &WgpuBuffer<T>,
        right: &WgpuBuffer<T>,
    ) -> Result<T> {
        Self::require_same_allocation("dot left operand", &prepared.left, left)?;
        Self::require_same_allocation("dot right operand", &prepared.right, right)?;
        prepared.operation.dispatch(device)?;
        Self::download_scalar(device, prepared.operation.output())
    }

    fn prepare_norm_l2<'a>(
        &self,
        device: &WgpuDevice,
        vector: &'a WgpuBuffer<T>,
    ) -> Result<Self::PreparedNorm<'a>> {
        let layout = Self::layout(vector.len())?;
        Ok(WgpuPreparedNorm {
            operation: prepare_norm_l2(
                device,
                StridedOperand {
                    buffer: vector,
                    layout: &layout,
                },
            )?,
            input: vector.clone(),
        })
    }

    fn norm_l1(&self, device: &WgpuDevice, vector: &WgpuBuffer<T>) -> Result<T> {
        let layout = leto::Layout::c_contiguous([vector.len()]).map_err(|error| {
            hephaestus_core::HephaestusError::DispatchFailed {
                message: format!("norm operand layout rejected: {error}"),
            }
        })?;
        let result = crate::application::linalg::norm_l1::<T, 1>(
            device,
            crate::application::strided::StridedOperand {
                buffer: vector,
                layout: &layout,
            },
        )?;
        Self::download_scalar(device, &result)
    }

    fn norm_max(&self, device: &WgpuDevice, vector: &WgpuBuffer<T>) -> Result<T> {
        let layout = leto::Layout::c_contiguous([vector.len()]).map_err(|error| {
            hephaestus_core::HephaestusError::DispatchFailed {
                message: format!("norm operand layout rejected: {error}"),
            }
        })?;
        let result = crate::application::linalg::norm_max::<T, 1>(
            device,
            crate::application::strided::StridedOperand {
                buffer: vector,
                layout: &layout,
            },
        )?;
        Self::download_scalar(device, &result)
    }

    fn norm_l2_prepared<'a>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedNorm<'a>,
        vector: &WgpuBuffer<T>,
    ) -> Result<T> {
        Self::require_same_allocation("norm operand", &prepared.input, vector)?;
        prepared.operation.dispatch(device)?;
        Self::download_scalar(device, prepared.operation.output())
    }
}

impl<T> RetainedReductions<WgpuDevice, T> for WgpuVectorOps<T>
where
    T: VectorScalar<Wgsl>
        + VectorFactor<Wgsl>
        + L2NormScalar
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, Wgsl>,
{
    // The WGPU prepared forms already own their operands: `WgpuBuffer` is a
    // shared handle, so cloning it keeps the allocation alive without copying
    // device memory. The retained and borrowing forms are therefore the same
    // resources reached through different lifetimes.
    type RetainedDot = WgpuPreparedDot<T>;
    type RetainedNorm = WgpuPreparedNorm<T>;

    fn retain_dot(
        &self,
        device: &WgpuDevice,
        left: &WgpuBuffer<T>,
        right: &WgpuBuffer<T>,
    ) -> Result<Self::RetainedDot> {
        self.prepare_dot(device, left, right)
    }

    fn dot_retained(
        &self,
        device: &WgpuDevice,
        retained: &Self::RetainedDot,
        left: &WgpuBuffer<T>,
        right: &WgpuBuffer<T>,
    ) -> Result<T> {
        self.dot_prepared(device, retained, left, right)
    }

    fn retain_norm_l2(
        &self,
        device: &WgpuDevice,
        vector: &WgpuBuffer<T>,
    ) -> Result<Self::RetainedNorm> {
        self.prepare_norm_l2(device, vector)
    }

    fn norm_l2_retained(
        &self,
        device: &WgpuDevice,
        retained: &Self::RetainedNorm,
        vector: &WgpuBuffer<T>,
    ) -> Result<T> {
        self.norm_l2_prepared(device, retained, vector)
    }
}
