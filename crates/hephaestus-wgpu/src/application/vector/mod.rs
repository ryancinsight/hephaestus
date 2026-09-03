//! WGPU implementation of the dense vector-operation seam.

mod kernels;

use eunomia::Pod;
use hephaestus_core::RetainedReductions;
use hephaestus_core::{
    Binding, CommandStream, ComputeDevice, DenseVectorOps, DeviceBuffer, DispatchGrid,
    HephaestusError, KernelDevice, Result,
};

use hephaestus_core::{AddOp, BlockWidth, DivOp, MulOp, SubOp};
use leto::Layout;

use crate::{
    L2NormScalar, PreparedDot, PreparedL2Norm, StridedOperand, WgpuBuffer, WgpuDevice,
    WgpuPrepared, binary_elementwise_into, prepare_dot, prepare_norm_l2,
};

use kernels::{AxpyKernel, ScaleKernel, VectorParams, XpayKernel, workgroup_count};

/// Prepared in-place vector kernels for one WGPU device.
///
/// Shader compilation happens once here rather than per operation, which is
/// what makes the seam usable inside an iteration loop.
pub struct WgpuVectorOps {
    scale: WgpuPrepared<ScaleKernel>,
    axpy: WgpuPrepared<AxpyKernel>,
    xpay: WgpuPrepared<XpayKernel>,
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

impl WgpuVectorOps {
    /// Compile the in-place vector kernels on `device`.
    ///
    /// # Errors
    ///
    /// Returns a shader preparation or device failure.
    pub fn new(device: &WgpuDevice) -> Result<Self> {
        Ok(Self {
            scale: device.prepare(&ScaleKernel)?,
            axpy: device.prepare(&AxpyKernel)?,
            xpay: device.prepare(&XpayKernel)?,
        })
    }

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
    fn require_same_allocation<T: Pod>(
        role: &str,
        expected: &WgpuBuffer<T>,
        actual: &WgpuBuffer<T>,
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

    fn dispatch_binary(
        &self,
        device: &WgpuDevice,
        prepared: &WgpuPrepared<AxpyKernel>,
        target: &WgpuBuffer<f32>,
        source: &WgpuBuffer<f32>,
        factor: f32,
    ) -> Result<()> {
        device.dispatch(
            prepared,
            &[Binding::read_write(target), Binding::read(source)],
            &VectorParams::new(factor, target.len())?,
            Self::grid(target.len())?,
        )
    }

    fn download_scalar<T: L2NormScalar>(device: &WgpuDevice, scalar: &WgpuBuffer<T>) -> Result<T> {
        let mut host = [T::zeroed(); 1];
        device.download(scalar, &mut host)?;
        Ok(host[0])
    }
}

impl DenseVectorOps<WgpuDevice, f32> for WgpuVectorOps {
    type PreparedDot<'a>
        = WgpuPreparedDot<f32>
    where
        Self: 'a;
    type PreparedNorm<'a>
        = WgpuPreparedNorm<f32>
    where
        Self: 'a;

    fn copy_vector(
        &self,
        device: &WgpuDevice,
        source: &WgpuBuffer<f32>,
        target: &WgpuBuffer<f32>,
    ) -> Result<()> {
        Self::require_equal_lengths(source.len(), target.len())?;
        if source.is_empty() {
            return Ok(());
        }
        let mut stream = device.stream()?;
        stream.copy(source, target)?;
        stream.submit()
    }

    fn scale_vector(
        &self,
        device: &WgpuDevice,
        target: &WgpuBuffer<f32>,
        factor: f32,
    ) -> Result<()> {
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
        target: &WgpuBuffer<f32>,
        source: &WgpuBuffer<f32>,
        factor: f32,
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
        target: &WgpuBuffer<f32>,
        source: &WgpuBuffer<f32>,
        factor: f32,
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
        left: &WgpuBuffer<f32>,
        right: &WgpuBuffer<f32>,
        output: &WgpuBuffer<f32>,
    ) -> Result<()> {
        Self::require_equal_lengths(left.len(), right.len())?;
        Self::require_equal_lengths(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<SubOp, f32>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn add_into(
        &self,
        device: &WgpuDevice,
        left: &WgpuBuffer<f32>,
        right: &WgpuBuffer<f32>,
        output: &WgpuBuffer<f32>,
    ) -> Result<()> {
        Self::require_equal_lengths(left.len(), right.len())?;
        Self::require_equal_lengths(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<AddOp, f32>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn multiply_into(
        &self,
        device: &WgpuDevice,
        left: &WgpuBuffer<f32>,
        right: &WgpuBuffer<f32>,
        output: &WgpuBuffer<f32>,
    ) -> Result<()> {
        Self::require_equal_lengths(left.len(), right.len())?;
        Self::require_equal_lengths(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<MulOp, f32>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn divide_into(
        &self,
        device: &WgpuDevice,
        left: &WgpuBuffer<f32>,
        right: &WgpuBuffer<f32>,
        output: &WgpuBuffer<f32>,
    ) -> Result<()> {
        Self::require_equal_lengths(left.len(), right.len())?;
        Self::require_equal_lengths(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<DivOp, f32>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn prepare_dot<'a>(
        &self,
        device: &WgpuDevice,
        left: &'a WgpuBuffer<f32>,
        right: &'a WgpuBuffer<f32>,
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
        left: &WgpuBuffer<f32>,
        right: &WgpuBuffer<f32>,
    ) -> Result<f32> {
        Self::require_same_allocation("dot left operand", &prepared.left, left)?;
        Self::require_same_allocation("dot right operand", &prepared.right, right)?;
        prepared.operation.dispatch(device)?;
        Self::download_scalar(device, prepared.operation.output())
    }

    fn prepare_norm_l2<'a>(
        &self,
        device: &WgpuDevice,
        vector: &'a WgpuBuffer<f32>,
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

    fn norm_l1(&self, device: &WgpuDevice, vector: &WgpuBuffer<f32>) -> Result<f32> {
        let layout = leto::Layout::c_contiguous([vector.len()]).map_err(|error| {
            hephaestus_core::HephaestusError::DispatchFailed {
                message: format!("norm operand layout rejected: {error}"),
            }
        })?;
        let result = crate::application::linalg::norm_l1::<f32, 1>(
            device,
            crate::application::strided::StridedOperand {
                buffer: vector,
                layout: &layout,
            },
        )?;
        Self::download_scalar(device, &result)
    }

    fn norm_max(&self, device: &WgpuDevice, vector: &WgpuBuffer<f32>) -> Result<f32> {
        let layout = leto::Layout::c_contiguous([vector.len()]).map_err(|error| {
            hephaestus_core::HephaestusError::DispatchFailed {
                message: format!("norm operand layout rejected: {error}"),
            }
        })?;
        let result = crate::application::linalg::norm_max::<f32, 1>(
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
        vector: &WgpuBuffer<f32>,
    ) -> Result<f32> {
        Self::require_same_allocation("norm operand", &prepared.input, vector)?;
        prepared.operation.dispatch(device)?;
        Self::download_scalar(device, prepared.operation.output())
    }
}

impl RetainedReductions<WgpuDevice, f32> for WgpuVectorOps {
    // The WGPU prepared forms already own their operands: `WgpuBuffer` is a
    // shared handle, so cloning it keeps the allocation alive without copying
    // device memory. The retained and borrowing forms are therefore the same
    // resources reached through different lifetimes.
    type RetainedDot = WgpuPreparedDot<f32>;
    type RetainedNorm = WgpuPreparedNorm<f32>;

    fn retain_dot(
        &self,
        device: &WgpuDevice,
        left: &WgpuBuffer<f32>,
        right: &WgpuBuffer<f32>,
    ) -> Result<Self::RetainedDot> {
        self.prepare_dot(device, left, right)
    }

    fn dot_retained(
        &self,
        device: &WgpuDevice,
        retained: &Self::RetainedDot,
        left: &WgpuBuffer<f32>,
        right: &WgpuBuffer<f32>,
    ) -> Result<f32> {
        self.dot_prepared(device, retained, left, right)
    }

    fn retain_norm_l2(
        &self,
        device: &WgpuDevice,
        vector: &WgpuBuffer<f32>,
    ) -> Result<Self::RetainedNorm> {
        self.prepare_norm_l2(device, vector)
    }

    fn norm_l2_retained(
        &self,
        device: &WgpuDevice,
        retained: &Self::RetainedNorm,
        vector: &WgpuBuffer<f32>,
    ) -> Result<f32> {
        self.norm_l2_prepared(device, retained, vector)
    }
}
