//! ROCm implementation of the backend-neutral dense vector-operation seam.

pub use crate::application::prepared_map_reduction::{PreparedDot, PreparedL2Norm};
use crate::application::prepared_map_reduction::{prepare_dense_dot, prepare_dense_norm_l2};
use crate::application::storage_kernel::{RocmMultiStorageKernel, RocmStorageBinding};
use crate::{AddOp, DivOp, MulOp, RocmBuffer, RocmDevice, binary_elementwise_into};
use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CommandStream, ComputeDevice, DenseVectorOps, DeviceBuffer, DispatchGrid,
    HephaestusError, KernelDevice, MultiStorageKernel, Result, SubOp,
};

const WORKGROUP_WIDTH: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VectorParams {
    len: u32,
    factor: f32,
}

impl VectorParams {
    fn new(len: usize, factor: f32) -> Result<Self> {
        Ok(Self {
            len: u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("dense vector length {len} exceeds u32::MAX"),
            })?,
            factor,
        })
    }
}

const SCALE_KERNEL: &str = r#"
struct VectorParams {
    unsigned int len;
    float factor;
};

extern "C" __global__ void scale_vector(float* target, VectorParams params) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < params.len) {
        target[index] = target[index] * params.factor;
    }
}
"#;

const AXPY_KERNEL: &str = r#"
struct VectorParams {
    unsigned int len;
    float factor;
};

extern "C" __global__ void axpy(float* target, const float* source, VectorParams params) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < params.len) {
        target[index] = fmaf(params.factor, source[index], target[index]);
    }
}
"#;

const XPAY_KERNEL: &str = r#"
struct VectorParams {
    unsigned int len;
    float factor;
};

extern "C" __global__ void xpay(float* target, const float* source, VectorParams params) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < params.len) {
        target[index] = fmaf(params.factor, target[index], source[index]);
    }
}
"#;

/// Prepared ROCm kernels for dense vector recurrences and reductions.
pub struct RocmVectorOps {
    scale: RocmMultiStorageKernel,
    axpy: RocmMultiStorageKernel,
    xpay: RocmMultiStorageKernel,
}

impl RocmVectorOps {
    /// Prepare dense vector kernels for a ROCm device.
    ///
    /// Kernel source is registered once; the HIP pipeline cache compiles each
    /// source on its first dispatch and reuses the resulting module thereafter.
    ///
    /// # Errors
    /// Returns a typed dispatch error when a kernel descriptor is invalid.
    pub fn new(_device: &RocmDevice) -> Result<Self> {
        Ok(Self {
            scale: RocmMultiStorageKernel::new(
                "hephaestus-rocm-vector-scale",
                SCALE_KERNEL,
                "scale_vector",
                &[0],
                [WORKGROUP_WIDTH, 1, 1],
                0,
            )?,
            axpy: RocmMultiStorageKernel::new(
                "hephaestus-rocm-vector-axpy",
                AXPY_KERNEL,
                "axpy",
                &[0, 1],
                [WORKGROUP_WIDTH, 1, 1],
                0,
            )?,
            xpay: RocmMultiStorageKernel::new(
                "hephaestus-rocm-vector-xpay",
                XPAY_KERNEL,
                "xpay",
                &[0, 1],
                [WORKGROUP_WIDTH, 1, 1],
                0,
            )?,
        })
    }

    fn lengths_match(left: usize, right: usize) -> Result<()> {
        if left == right {
            Ok(())
        } else {
            Err(HephaestusError::LengthMismatch {
                host_len: left,
                device_len: right,
            })
        }
    }

    fn grid(len: usize) -> Result<DispatchGrid> {
        DispatchGrid::covering_domain([len, 1, 1], [WORKGROUP_WIDTH as usize, 1, 1])
    }

    fn scalar(device: &RocmDevice, buffer: &RocmBuffer<f32>) -> Result<f32> {
        let mut output = [0.0_f32; 1];
        device.download(buffer, &mut output)?;
        Ok(output[0])
    }
}

impl DenseVectorOps<RocmDevice, f32> for RocmVectorOps {
    type PreparedDot<'a>
        = PreparedDot<'a, f32>
    where
        Self: 'a;
    type PreparedNorm<'a>
        = PreparedL2Norm<'a, f32, 1>
    where
        Self: 'a;

    fn copy_vector(
        &self,
        device: &RocmDevice,
        source: &RocmBuffer<f32>,
        target: &RocmBuffer<f32>,
    ) -> Result<()> {
        Self::lengths_match(source.len(), target.len())?;
        if source.is_empty() {
            return Ok(());
        }
        let mut stream = device.stream()?;
        stream.copy(source, target)?;
        stream.submit()
    }

    fn scale_vector(
        &self,
        device: &RocmDevice,
        target: &RocmBuffer<f32>,
        factor: f32,
    ) -> Result<()> {
        if target.is_empty() {
            return Ok(());
        }
        let params = VectorParams::new(target.len(), factor)?;
        MultiStorageKernel::<RocmDevice, VectorParams, [RocmStorageBinding<'_>; 1]>::dispatch(
            &self.scale,
            device,
            [RocmStorageBinding::new(0, target)],
            &params,
            Self::grid(target.len())?,
        )
    }

    fn axpy(
        &self,
        device: &RocmDevice,
        target: &RocmBuffer<f32>,
        source: &RocmBuffer<f32>,
        factor: f32,
    ) -> Result<()> {
        Self::lengths_match(target.len(), source.len())?;
        if target.is_empty() {
            return Ok(());
        }
        let params = VectorParams::new(target.len(), factor)?;
        MultiStorageKernel::<RocmDevice, VectorParams, [RocmStorageBinding<'_>; 2]>::dispatch(
            &self.axpy,
            device,
            [
                RocmStorageBinding::new(0, target),
                RocmStorageBinding::new(1, source),
            ],
            &params,
            Self::grid(target.len())?,
        )
    }

    fn xpay(
        &self,
        device: &RocmDevice,
        target: &RocmBuffer<f32>,
        source: &RocmBuffer<f32>,
        factor: f32,
    ) -> Result<()> {
        Self::lengths_match(target.len(), source.len())?;
        if target.is_empty() {
            return Ok(());
        }
        let params = VectorParams::new(target.len(), factor)?;
        MultiStorageKernel::<RocmDevice, VectorParams, [RocmStorageBinding<'_>; 2]>::dispatch(
            &self.xpay,
            device,
            [
                RocmStorageBinding::new(0, target),
                RocmStorageBinding::new(1, source),
            ],
            &params,
            Self::grid(target.len())?,
        )
    }

    fn subtract_into(
        &self,
        device: &RocmDevice,
        left: &RocmBuffer<f32>,
        right: &RocmBuffer<f32>,
        output: &RocmBuffer<f32>,
    ) -> Result<()> {
        Self::lengths_match(left.len(), right.len())?;
        Self::lengths_match(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<SubOp, f32>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn add_into(
        &self,
        device: &RocmDevice,
        left: &RocmBuffer<f32>,
        right: &RocmBuffer<f32>,
        output: &RocmBuffer<f32>,
    ) -> Result<()> {
        Self::lengths_match(left.len(), right.len())?;
        Self::lengths_match(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<AddOp, f32>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn multiply_into(
        &self,
        device: &RocmDevice,
        left: &RocmBuffer<f32>,
        right: &RocmBuffer<f32>,
        output: &RocmBuffer<f32>,
    ) -> Result<()> {
        Self::lengths_match(left.len(), right.len())?;
        Self::lengths_match(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<MulOp, f32>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn divide_into(
        &self,
        device: &RocmDevice,
        left: &RocmBuffer<f32>,
        right: &RocmBuffer<f32>,
        output: &RocmBuffer<f32>,
    ) -> Result<()> {
        Self::lengths_match(left.len(), right.len())?;
        Self::lengths_match(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<DivOp, f32>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn prepare_dot<'a>(
        &self,
        device: &RocmDevice,
        left: &'a RocmBuffer<f32>,
        right: &'a RocmBuffer<f32>,
    ) -> Result<Self::PreparedDot<'a>> {
        Self::lengths_match(left.len(), right.len())?;
        prepare_dense_dot(device, left, right)
    }

    fn dot_prepared<'a>(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedDot<'a>,
        left: &RocmBuffer<f32>,
        right: &RocmBuffer<f32>,
    ) -> Result<f32> {
        if !prepared.matches(left, right) {
            return Err(HephaestusError::DispatchFailed {
                message: "prepared dot received different device allocations".to_string(),
            });
        }
        prepared.dispatch()?;
        Self::scalar(device, prepared.output())
    }

    fn prepare_norm_l2<'a>(
        &self,
        device: &RocmDevice,
        vector: &'a RocmBuffer<f32>,
    ) -> Result<Self::PreparedNorm<'a>> {
        prepare_dense_norm_l2(device, vector)
    }

    fn norm_l1(&self, device: &RocmDevice, vector: &RocmBuffer<f32>) -> Result<f32> {
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
        Self::scalar(device, &result)
    }

    fn norm_max(&self, device: &RocmDevice, vector: &RocmBuffer<f32>) -> Result<f32> {
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
        Self::scalar(device, &result)
    }

    fn norm_l2_prepared<'a>(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedNorm<'a>,
        vector: &RocmBuffer<f32>,
    ) -> Result<f32> {
        if !prepared.matches(vector) {
            return Err(HephaestusError::DispatchFailed {
                message: "prepared norm received a different device allocation".to_string(),
            });
        }
        prepared.dispatch()?;
        Self::scalar(device, prepared.output())
    }
}
