//! ROCm implementation of the backend-neutral dense vector-operation seam.

pub use crate::application::prepared_map_reduction::{PreparedDot, PreparedL2Norm};
use crate::application::prepared_map_reduction::{prepare_dense_dot, prepare_dense_norm_l2};
use crate::application::storage_kernel::{RocmMultiStorageKernel, RocmStorageBinding};
use crate::{AddOp, DivOp, MulOp, RocmBuffer, RocmDevice, binary_elementwise_into};
use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CommandStream, ComputeDevice, DenseVectorOps, DeviceBuffer, DispatchGrid,
    HephaestusError, HipC, IdentityToken, KernelDevice, MaxOp, MultiStorageKernel, OpIdentity,
    Result, SubOp, SumOp, VectorScalar,
};
use std::marker::PhantomData;

const WORKGROUP_WIDTH: u32 = 256;

/// Scalar-independent uniform block for the vector kernels.
///
/// The factor rides widened to `f64`: lossless for `f32` (widening is exact
/// and the device narrowing round-trips), direct for `f64`. One
/// padding-free layout serves every scalar, so no per-scalar block exists.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VectorParams {
    len: u32,
    _pad: u32,
    factor: f64,
}

impl VectorParams {
    fn new<T: Into<f64>>(len: usize, factor: T) -> Result<Self> {
        Ok(Self {
            len: u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("dense vector length {len} exceeds u32::MAX"),
            })?,
            _pad: 0,
            factor: factor.into(),
        })
    }
}

fn params_struct<T: VectorScalar<HipC>>() -> String {
    format!(
        r#"{prelude}struct VectorParams {{
    unsigned int len;
    unsigned int _pad;
    double factor;
}};

"#,
        prelude = T::PRELUDE,
    )
}

fn scale_source<T: VectorScalar<HipC>>() -> String {
    format!(
        r#"{params}extern "C" __global__ void scale_vector({ty}* target, VectorParams params) {{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < params.len) {{
        {ty} f = ({ty})params.factor;
        target[index] = target[index] * f;
    }}
}}
"#,
        params = params_struct::<T>(),
        ty = T::TYPE_TOKEN,
    )
}

fn axpy_source<T: VectorScalar<HipC>>() -> String {
    let madd = T::madd_expr("f", "source[index]", "target[index]");
    format!(
        r#"{params}extern "C" __global__ void axpy({ty}* target, const {ty}* source, VectorParams params) {{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < params.len) {{
        {ty} f = ({ty})params.factor;
        target[index] = {madd};
    }}
}}
"#,
        params = params_struct::<T>(),
        ty = T::TYPE_TOKEN,
    )
}

fn xpay_source<T: VectorScalar<HipC>>() -> String {
    let madd = T::madd_expr("f", "target[index]", "source[index]");
    format!(
        r#"{params}extern "C" __global__ void xpay({ty}* target, const {ty}* source, VectorParams params) {{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < params.len) {{
        {ty} f = ({ty})params.factor;
        target[index] = {madd};
    }}
}}
"#,
        params = params_struct::<T>(),
        ty = T::TYPE_TOKEN,
    )
}

/// Prepared CUDA kernels for dense vector recurrences and reductions.
pub struct RocmVectorOps<T> {
    scale: RocmMultiStorageKernel,
    axpy: RocmMultiStorageKernel,
    xpay: RocmMultiStorageKernel,
    marker: PhantomData<T>,
}

impl<T: VectorScalar<HipC>> RocmVectorOps<T> {
    /// Prepare dense vector kernels for a ROCm device.
    ///
    /// Kernel source is registered once per scalar; the HIP pipeline cache
    /// compiles each source on its first dispatch and reuses the resulting
    /// module thereafter.
    ///
    /// # Errors
    /// Returns a typed dispatch error when a kernel descriptor is invalid.
    pub fn new(_device: &RocmDevice) -> Result<Self> {
        // Intentionally leaked: generated sources live as long as the
        // pipeline-cache entries compiled from them, i.e. the process.
        fn leaked(source: String) -> &'static str {
            Box::leak(source.into_boxed_str())
        }
        Ok(Self {
            scale: RocmMultiStorageKernel::new(
                "hephaestus-rocm-vector-scale",
                leaked(scale_source::<T>()),
                "scale_vector",
                &[0],
                [WORKGROUP_WIDTH, 1, 1],
                0,
            )?,
            axpy: RocmMultiStorageKernel::new(
                "hephaestus-rocm-vector-axpy",
                leaked(axpy_source::<T>()),
                "axpy",
                &[0, 1],
                [WORKGROUP_WIDTH, 1, 1],
                0,
            )?,
            xpay: RocmMultiStorageKernel::new(
                "hephaestus-rocm-vector-xpay",
                leaked(xpay_source::<T>()),
                "xpay",
                &[0, 1],
                [WORKGROUP_WIDTH, 1, 1],
                0,
            )?,
            marker: PhantomData,
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
}

impl<T> RocmVectorOps<T>
where
    T: VectorScalar<HipC> + Zeroable + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    fn scalar(device: &RocmDevice, buffer: &RocmBuffer<T>) -> Result<T> {
        let mut output = [T::zeroed(); 1];
        device.download(buffer, &mut output)?;
        Ok(output[0])
    }

    fn require_dot_operands<'a>(
        device: &RocmDevice,
        left: &'a RocmBuffer<T>,
        right: &'a RocmBuffer<T>,
    ) -> Result<PreparedDot<'a, T>> {
        Self::lengths_match(left.len(), right.len())?;
        prepare_dense_dot(device, left, right)
    }
}

impl<T> DenseVectorOps<RocmDevice, T> for RocmVectorOps<T>
where
    T: VectorScalar<HipC>
        + Into<f64>
        + Zeroable
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, HipC>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, HipC>,
{
    type PreparedDot<'a>
        = PreparedDot<'a, T>
    where
        Self: 'a;
    type PreparedNorm<'a>
        = PreparedL2Norm<'a, T, 1>
    where
        Self: 'a;

    fn copy_vector(
        &self,
        device: &RocmDevice,
        source: &RocmBuffer<T>,
        target: &RocmBuffer<T>,
    ) -> Result<()> {
        Self::lengths_match(source.len(), target.len())?;
        if source.is_empty() {
            return Ok(());
        }
        let mut stream = device.stream()?;
        stream.copy(source, target)?;
        stream.submit()
    }

    fn scale_vector(&self, device: &RocmDevice, target: &RocmBuffer<T>, factor: T) -> Result<()> {
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
        target: &RocmBuffer<T>,
        source: &RocmBuffer<T>,
        factor: T,
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
        target: &RocmBuffer<T>,
        source: &RocmBuffer<T>,
        factor: T,
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
        left: &RocmBuffer<T>,
        right: &RocmBuffer<T>,
        output: &RocmBuffer<T>,
    ) -> Result<()> {
        Self::lengths_match(left.len(), right.len())?;
        Self::lengths_match(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<SubOp, T>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn add_into(
        &self,
        device: &RocmDevice,
        left: &RocmBuffer<T>,
        right: &RocmBuffer<T>,
        output: &RocmBuffer<T>,
    ) -> Result<()> {
        Self::lengths_match(left.len(), right.len())?;
        Self::lengths_match(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<AddOp, T>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn multiply_into(
        &self,
        device: &RocmDevice,
        left: &RocmBuffer<T>,
        right: &RocmBuffer<T>,
        output: &RocmBuffer<T>,
    ) -> Result<()> {
        Self::lengths_match(left.len(), right.len())?;
        Self::lengths_match(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<MulOp, T>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn divide_into(
        &self,
        device: &RocmDevice,
        left: &RocmBuffer<T>,
        right: &RocmBuffer<T>,
        output: &RocmBuffer<T>,
    ) -> Result<()> {
        Self::lengths_match(left.len(), right.len())?;
        Self::lengths_match(left.len(), output.len())?;
        if left.is_empty() {
            return Ok(());
        }
        binary_elementwise_into::<DivOp, T>(device, left, right, output, BlockWidth::DEFAULT)
    }

    fn prepare_dot<'a>(
        &self,
        device: &RocmDevice,
        left: &'a RocmBuffer<T>,
        right: &'a RocmBuffer<T>,
    ) -> Result<Self::PreparedDot<'a>> {
        Self::require_dot_operands(device, left, right)
    }

    fn dot_prepared<'a>(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedDot<'a>,
        left: &RocmBuffer<T>,
        right: &RocmBuffer<T>,
    ) -> Result<T> {
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
        vector: &'a RocmBuffer<T>,
    ) -> Result<Self::PreparedNorm<'a>> {
        prepare_dense_norm_l2(device, vector)
    }

    fn norm_l1(&self, device: &RocmDevice, vector: &RocmBuffer<T>) -> Result<T> {
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
        Self::scalar(device, &result)
    }

    fn norm_max(&self, device: &RocmDevice, vector: &RocmBuffer<T>) -> Result<T> {
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
        Self::scalar(device, &result)
    }

    fn norm_l2_prepared<'a>(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedNorm<'a>,
        vector: &RocmBuffer<T>,
    ) -> Result<T> {
        if !prepared.matches(vector) {
            return Err(HephaestusError::DispatchFailed {
                message: "prepared norm received a different device allocation".to_string(),
            });
        }
        prepared.dispatch()?;
        Self::scalar(device, prepared.output())
    }
}
