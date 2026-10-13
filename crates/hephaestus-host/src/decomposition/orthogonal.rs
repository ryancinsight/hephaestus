use hephaestus_core::{BidiagonalHandle, HessenbergHandle, SchurHandle};

use crate::{HostBuffer, HostDevice};

/// Real Schur form on the host.
pub struct HostSchur {
    pub(crate) order: usize,
    pub(crate) q: HostBuffer<f32>,
    pub(crate) t: HostBuffer<f32>,
}

impl SchurHandle<HostDevice> for HostSchur {
    fn order(&self) -> usize {
        self.order
    }
    fn q_buffer(&self) -> &HostBuffer<f32> {
        &self.q
    }
    fn t_buffer(&self) -> &HostBuffer<f32> {
        &self.t
    }
}

/// Hessenberg reduction on the host.
pub struct HostHessenberg {
    pub(crate) order: usize,
    pub(crate) q: HostBuffer<f32>,
    pub(crate) h: HostBuffer<f32>,
}

impl HessenbergHandle<HostDevice> for HostHessenberg {
    fn order(&self) -> usize {
        self.order
    }
    fn q_buffer(&self) -> &HostBuffer<f32> {
        &self.q
    }
    fn h_buffer(&self) -> &HostBuffer<f32> {
        &self.h
    }
}

/// Bidiagonal reduction on the host.
pub struct HostBidiagonal {
    pub(crate) shape: (usize, usize),
    pub(crate) u: HostBuffer<f32>,
    pub(crate) b: HostBuffer<f32>,
    pub(crate) v: HostBuffer<f32>,
}

impl BidiagonalHandle<HostDevice> for HostBidiagonal {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }
    fn u_buffer(&self) -> &HostBuffer<f32> {
        &self.u
    }
    fn b_buffer(&self) -> &HostBuffer<f32> {
        &self.b
    }
    fn v_buffer(&self) -> &HostBuffer<f32> {
        &self.v
    }
}
