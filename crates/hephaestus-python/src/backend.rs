//! Backend bridging: device/buffer/CSR enums spanning the wgpu and CUDA
//! backends, plus the dispatch macros the operation modules expand to
//! route a Python call to the matching backend kernel.

use eunomia::Complex;
use hephaestus_core::{ComputeDevice, DeviceBuffer};
use hephaestus_cuda::{CudaBuffer, CudaDevice};
use hephaestus_wgpu::{WgpuBuffer, WgpuDevice};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

// Size skew (WgpuDevice ~312 B vs CudaDevice ~72 B, dominated by cached
// adapter info/limits) is irrelevant here: a BackendDevice is constructed
// once per Python device object and matched per call, never stored in bulk.
// Boxing would add an indirection to every dispatch for no measurable win.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub(crate) enum BackendDevice {
    Wgpu(WgpuDevice),
    Cuda(CudaDevice),
}

impl BackendDevice {
    pub(crate) fn try_default(backend: Option<&str>) -> PyResult<Self> {
        match backend.unwrap_or("wgpu").to_ascii_lowercase().as_str() {
            "wgpu" => WgpuDevice::try_default("hephaestus-py-device")
                .map(Self::Wgpu)
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
            "cuda" => CudaDevice::try_default()
                .map(Self::Cuda)
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
            other => Err(PyValueError::new_err(format!(
                "unsupported backend {other:?}; expected 'wgpu' or 'cuda'"
            ))),
        }
    }

    pub(crate) fn backend_name(&self) -> &'static str {
        match self {
            Self::Wgpu(device) => device.backend_name(),
            Self::Cuda(device) => device.backend_name(),
        }
    }

    pub(crate) fn alloc_zeroed_f32(&self, len: usize) -> hephaestus_core::Result<BackendBuffer> {
        match self {
            Self::Wgpu(device) => device.alloc_zeroed::<f32>(len).map(BackendBuffer::Wgpu),
            Self::Cuda(device) => device
                .alloc_zeroed::<f32>(len)
                .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer))),
        }
    }

    pub(crate) fn upload_f32(&self, data: &[f32]) -> hephaestus_core::Result<BackendBuffer> {
        match self {
            Self::Wgpu(device) => device.upload(data).map(BackendBuffer::Wgpu),
            Self::Cuda(device) => device
                .upload(data)
                .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer))),
        }
    }

    pub(crate) fn download_f32(
        &self,
        buffer: &BackendBuffer,
        out: &mut [f32],
    ) -> hephaestus_core::Result<()> {
        match (self, buffer) {
            (Self::Wgpu(device), BackendBuffer::Wgpu(buffer)) => device.download(buffer, out),
            (Self::Cuda(device), BackendBuffer::Cuda(buffer)) => device.download(buffer, out),
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        }
    }

    pub(crate) fn download_complex(
        &self,
        buffer: &BackendComplexBuffer,
        out: &mut [Complex<f32>],
    ) -> hephaestus_core::Result<()> {
        match (self, buffer) {
            (Self::Wgpu(device), BackendComplexBuffer::Wgpu(buffer)) => {
                device.download(buffer, out)
            }
            (Self::Cuda(device), BackendComplexBuffer::Cuda(buffer)) => {
                device.download(buffer, out)
            }
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        }
    }
}

#[derive(Clone)]
pub(crate) enum BackendBuffer {
    Wgpu(WgpuBuffer<f32>),
    Cuda(Arc<CudaBuffer<f32>>),
}

impl BackendBuffer {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Wgpu(buffer) => buffer.len(),
            Self::Cuda(buffer) => buffer.len(),
        }
    }
}

pub(crate) enum BackendComplexBuffer {
    Wgpu(WgpuBuffer<Complex<f32>>),
    Cuda(CudaBuffer<Complex<f32>>),
}

#[derive(Clone)]
pub(crate) enum BackendCsrMatrix {
    Wgpu(hephaestus_wgpu::GpuCsrMatrix<f32>),
    Cuda(Arc<hephaestus_cuda::GpuCsrMatrix<f32>>),
}

impl BackendCsrMatrix {
    pub(crate) fn shape(&self) -> (usize, usize) {
        match self {
            Self::Wgpu(matrix) => matrix.shape(),
            Self::Cuda(matrix) => matrix.shape(),
        }
    }

    pub(crate) fn nnz(&self) -> usize {
        match self {
            Self::Wgpu(matrix) => matrix.nnz(),
            Self::Cuda(matrix) => matrix.nnz(),
        }
    }
}

macro_rules! backend_unary {
    ($device:expr, $buffer:expr, $wgpu_op:ty, $cuda_op:ty) => {
        match ($device, $buffer) {
            (
                $crate::backend::BackendDevice::Wgpu(device),
                $crate::backend::BackendBuffer::Wgpu(buffer),
            ) => hephaestus_wgpu::unary_elementwise::<$wgpu_op, f32>(device, buffer)
                .map($crate::backend::BackendBuffer::Wgpu),
            (
                $crate::backend::BackendDevice::Cuda(device),
                $crate::backend::BackendBuffer::Cuda(buffer),
            ) => hephaestus_cuda::unary_elementwise::<$cuda_op, f32>(device, buffer)
                .map(|buffer| $crate::backend::BackendBuffer::Cuda(std::sync::Arc::new(buffer))),
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array operands belong to different backends".to_string(),
            }),
        }
    };
}

macro_rules! backend_binary {
    ($device:expr, $lhs:expr, $rhs:expr, $wgpu_op:ty, $cuda_op:ty) => {
        match ($device, $lhs, $rhs) {
            (
                $crate::backend::BackendDevice::Wgpu(device),
                $crate::backend::BackendBuffer::Wgpu(lhs),
                $crate::backend::BackendBuffer::Wgpu(rhs),
            ) => hephaestus_wgpu::binary_elementwise::<$wgpu_op, f32>(device, lhs, rhs)
                .map($crate::backend::BackendBuffer::Wgpu),
            (
                $crate::backend::BackendDevice::Cuda(device),
                $crate::backend::BackendBuffer::Cuda(lhs),
                $crate::backend::BackendBuffer::Cuda(rhs),
            ) => hephaestus_cuda::binary_elementwise::<$cuda_op, f32>(device, lhs, rhs)
                .map(|buffer| $crate::backend::BackendBuffer::Cuda(std::sync::Arc::new(buffer))),
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array operands belong to different backends".to_string(),
            }),
        }
    };
}

macro_rules! backend_scalar {
    ($device:expr, $buffer:expr, $scalar:expr, $wgpu_op:ty, $cuda_op:ty) => {
        match ($device, $buffer) {
            (
                $crate::backend::BackendDevice::Wgpu(device),
                $crate::backend::BackendBuffer::Wgpu(buffer),
            ) => hephaestus_wgpu::scalar_elementwise::<$wgpu_op, f32>(device, buffer, $scalar)
                .map($crate::backend::BackendBuffer::Wgpu),
            (
                $crate::backend::BackendDevice::Cuda(device),
                $crate::backend::BackendBuffer::Cuda(buffer),
            ) => hephaestus_cuda::scalar_elementwise::<$cuda_op, f32>(device, buffer, $scalar)
                .map(|buffer| $crate::backend::BackendBuffer::Cuda(std::sync::Arc::new(buffer))),
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        }
    };
}

macro_rules! backend_reduction {
    ($device:expr, $buffer:expr, $wgpu_op:ty, $cuda_op:ty) => {
        match ($device, $buffer) {
            (
                $crate::backend::BackendDevice::Wgpu(device),
                $crate::backend::BackendBuffer::Wgpu(buffer),
            ) => hephaestus_wgpu::reduction::<$wgpu_op, f32>(device, buffer)
                .map($crate::backend::BackendBuffer::Wgpu),
            (
                $crate::backend::BackendDevice::Cuda(device),
                $crate::backend::BackendBuffer::Cuda(buffer),
            ) => hephaestus_cuda::reduction::<$cuda_op, f32>(device, buffer)
                .map(|buffer| $crate::backend::BackendBuffer::Cuda(std::sync::Arc::new(buffer))),
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        }
    };
}

macro_rules! backend_norm {
    ($device:expr, $buffer:expr, $layout:expr, $wgpu_fn:path, $cuda_fn:path) => {
        match ($device, $buffer) {
            (
                $crate::backend::BackendDevice::Wgpu(device),
                $crate::backend::BackendBuffer::Wgpu(buffer),
            ) => $wgpu_fn(
                device,
                hephaestus_wgpu::StridedOperand {
                    buffer,
                    layout: &$layout,
                },
            )
            .map($crate::backend::BackendBuffer::Wgpu),
            (
                $crate::backend::BackendDevice::Cuda(device),
                $crate::backend::BackendBuffer::Cuda(buffer),
            ) => $cuda_fn(
                device,
                hephaestus_cuda::StridedOperand {
                    buffer,
                    layout: &$layout,
                },
            )
            .map(|buffer| $crate::backend::BackendBuffer::Cuda(std::sync::Arc::new(buffer))),
            _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "array buffer belongs to a different backend".to_string(),
            }),
        }
    };
}

pub(crate) use backend_binary;
pub(crate) use backend_norm;
pub(crate) use backend_reduction;
pub(crate) use backend_scalar;
pub(crate) use backend_unary;

/// Clone a CUDA buffer via a full host round-trip (download then upload).
///
/// Cost: two host<->device transfers of `buffer.len() * 4` bytes each,
/// staged through host memory. Superseded by a device-side copy when
/// `CommandStream::copy` lands in the CUDA backend.
pub(crate) fn clone_cuda_buffer(
    device: &CudaDevice,
    buffer: &CudaBuffer<f32>,
) -> hephaestus_core::Result<BackendBuffer> {
    let mut host = vec![0.0f32; buffer.len()];
    device.download(buffer, &mut host)?;
    device
        .upload(&host)
        .map(|buffer| BackendBuffer::Cuda(Arc::new(buffer)))
}
