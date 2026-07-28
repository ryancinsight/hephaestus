//! CUDA implementation of the backend-neutral dense vector-operation seam.

pub use crate::application::prepared_map_reduction::{PreparedDot, PreparedL2Norm};
use crate::application::prepared_map_reduction::{prepare_dense_dot, prepare_dense_norm_l2};
use crate::application::storage_kernel::{CudaMultiStorageKernel, CudaStorageBinding};
use crate::{AddOp, CudaBuffer, CudaDevice, DivOp, MulOp, binary_elementwise_into};
use bytemuck::{Pod, Zeroable};
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

/// Prepared CUDA kernels for dense vector recurrences and reductions.
pub struct CudaVectorOps {
    scale: CudaMultiStorageKernel,
    axpy: CudaMultiStorageKernel,
    xpay: CudaMultiStorageKernel,
}

impl CudaVectorOps {
    /// Prepare dense vector kernels for a CUDA device.
    ///
    /// Kernel source is registered once; the CUDA pipeline cache compiles each
    /// source on its first dispatch and reuses the resulting module thereafter.
    ///
    /// # Errors
    /// Returns a typed dispatch error when a kernel descriptor is invalid.
    pub fn new(_device: &CudaDevice) -> Result<Self> {
        Ok(Self {
            scale: CudaMultiStorageKernel::new(
                "hephaestus-cuda-vector-scale",
                SCALE_KERNEL,
                "scale_vector",
                &[0],
                [WORKGROUP_WIDTH, 1, 1],
                0,
            )?,
            axpy: CudaMultiStorageKernel::new(
                "hephaestus-cuda-vector-axpy",
                AXPY_KERNEL,
                "axpy",
                &[0, 1],
                [WORKGROUP_WIDTH, 1, 1],
                0,
            )?,
            xpay: CudaMultiStorageKernel::new(
                "hephaestus-cuda-vector-xpay",
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

    fn scalar(device: &CudaDevice, buffer: &CudaBuffer<f32>) -> Result<f32> {
        let mut output = [0.0_f32; 1];
        device.download(buffer, &mut output)?;
        Ok(output[0])
    }

    fn require_dot_operands<'a>(
        device: &CudaDevice,
        left: &'a CudaBuffer<f32>,
        right: &'a CudaBuffer<f32>,
    ) -> Result<PreparedDot<'a, f32>> {
        Self::lengths_match(left.len(), right.len())?;
        prepare_dense_dot(device, left, right)
    }
}

impl DenseVectorOps<CudaDevice, f32> for CudaVectorOps {
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
        device: &CudaDevice,
        source: &CudaBuffer<f32>,
        target: &CudaBuffer<f32>,
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
        device: &CudaDevice,
        target: &CudaBuffer<f32>,
        factor: f32,
    ) -> Result<()> {
        if target.is_empty() {
            return Ok(());
        }
        let params = VectorParams::new(target.len(), factor)?;
        MultiStorageKernel::<CudaDevice, VectorParams, [CudaStorageBinding<'_>; 1]>::dispatch(
            &self.scale,
            device,
            [CudaStorageBinding::new(0, target)],
            &params,
            Self::grid(target.len())?,
        )
    }

    fn axpy(
        &self,
        device: &CudaDevice,
        target: &CudaBuffer<f32>,
        source: &CudaBuffer<f32>,
        factor: f32,
    ) -> Result<()> {
        Self::lengths_match(target.len(), source.len())?;
        if target.is_empty() {
            return Ok(());
        }
        let params = VectorParams::new(target.len(), factor)?;
        MultiStorageKernel::<CudaDevice, VectorParams, [CudaStorageBinding<'_>; 2]>::dispatch(
            &self.axpy,
            device,
            [
                CudaStorageBinding::new(0, target),
                CudaStorageBinding::new(1, source),
            ],
            &params,
            Self::grid(target.len())?,
        )
    }

    fn xpay(
        &self,
        device: &CudaDevice,
        target: &CudaBuffer<f32>,
        source: &CudaBuffer<f32>,
        factor: f32,
    ) -> Result<()> {
        Self::lengths_match(target.len(), source.len())?;
        if target.is_empty() {
            return Ok(());
        }
        let params = VectorParams::new(target.len(), factor)?;
        MultiStorageKernel::<CudaDevice, VectorParams, [CudaStorageBinding<'_>; 2]>::dispatch(
            &self.xpay,
            device,
            [
                CudaStorageBinding::new(0, target),
                CudaStorageBinding::new(1, source),
            ],
            &params,
            Self::grid(target.len())?,
        )
    }

    fn subtract_into(
        &self,
        device: &CudaDevice,
        left: &CudaBuffer<f32>,
        right: &CudaBuffer<f32>,
        output: &CudaBuffer<f32>,
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
        device: &CudaDevice,
        left: &CudaBuffer<f32>,
        right: &CudaBuffer<f32>,
        output: &CudaBuffer<f32>,
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
        device: &CudaDevice,
        left: &CudaBuffer<f32>,
        right: &CudaBuffer<f32>,
        output: &CudaBuffer<f32>,
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
        device: &CudaDevice,
        left: &CudaBuffer<f32>,
        right: &CudaBuffer<f32>,
        output: &CudaBuffer<f32>,
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
        device: &CudaDevice,
        left: &'a CudaBuffer<f32>,
        right: &'a CudaBuffer<f32>,
    ) -> Result<Self::PreparedDot<'a>> {
        Self::require_dot_operands(device, left, right)
    }

    fn dot_prepared<'a>(
        &self,
        device: &CudaDevice,
        prepared: &Self::PreparedDot<'a>,
        left: &CudaBuffer<f32>,
        right: &CudaBuffer<f32>,
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
        device: &CudaDevice,
        vector: &'a CudaBuffer<f32>,
    ) -> Result<Self::PreparedNorm<'a>> {
        prepare_dense_norm_l2(device, vector)
    }

    fn norm_l2_prepared<'a>(
        &self,
        device: &CudaDevice,
        prepared: &Self::PreparedNorm<'a>,
        vector: &CudaBuffer<f32>,
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
