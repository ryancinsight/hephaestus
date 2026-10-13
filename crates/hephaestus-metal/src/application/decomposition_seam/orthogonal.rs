use hephaestus_core::{BidiagonalHandle, HessenbergHandle, SchurHandle};

use crate::application::decomposition::{
    GpuBidiagonalDecomposition, GpuHessenbergDecomposition, GpuRealSchur,
};
use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Real Schur result: the WGPU handle plus its factors rewrapped for
/// this device.
pub struct MetalRealSchur {
    pub(crate) inner: GpuRealSchur,
    pub(crate) q: MetalBuffer<f32>,
    pub(crate) t: MetalBuffer<f32>,
}

impl SchurHandle<MetalDevice> for MetalRealSchur {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn q_buffer(&self) -> &MetalBuffer<f32> {
        &self.q
    }
    fn t_buffer(&self) -> &MetalBuffer<f32> {
        &self.t
    }
}

/// Hessenberg result: the WGPU handle plus its factors rewrapped for
/// this device.
pub struct MetalHessenbergDecomposition {
    pub(crate) inner: GpuHessenbergDecomposition,
    pub(crate) q: MetalBuffer<f32>,
    pub(crate) h: MetalBuffer<f32>,
}

impl HessenbergHandle<MetalDevice> for MetalHessenbergDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn q_buffer(&self) -> &MetalBuffer<f32> {
        &self.q
    }
    fn h_buffer(&self) -> &MetalBuffer<f32> {
        &self.h
    }
}

/// Bidiagonal result: the WGPU handle plus its factors rewrapped for
/// this device.
pub struct MetalBidiagonalDecomposition {
    pub(crate) inner: GpuBidiagonalDecomposition,
    pub(crate) u: MetalBuffer<f32>,
    pub(crate) b: MetalBuffer<f32>,
    pub(crate) v: MetalBuffer<f32>,
}

impl BidiagonalHandle<MetalDevice> for MetalBidiagonalDecomposition {
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    fn u_buffer(&self) -> &MetalBuffer<f32> {
        &self.u
    }
    fn b_buffer(&self) -> &MetalBuffer<f32> {
        &self.b
    }
    fn v_buffer(&self) -> &MetalBuffer<f32> {
        &self.v
    }
}
