use hephaestus_core::{SvdHandle, SymmetricEigenHandle};

use crate::{HostBuffer, HostDevice};

/// Symmetric Jacobi eigendecomposition on the host.
pub struct HostSymmetricEigen {
    pub(crate) order: usize,
    pub(crate) eigenvalues: HostBuffer<f32>,
    pub(crate) eigenvectors: HostBuffer<f32>,
}

impl SymmetricEigenHandle<HostDevice> for HostSymmetricEigen {
    fn order(&self) -> usize {
        self.order
    }
    fn eigenvalues(&self) -> &HostBuffer<f32> {
        &self.eigenvalues
    }
    fn eigenvectors(&self) -> &HostBuffer<f32> {
        &self.eigenvectors
    }
}

/// SVD on the host.
pub struct HostSvd {
    pub(crate) shape: (usize, usize),
    pub(crate) u: HostBuffer<f32>,
    pub(crate) v: HostBuffer<f32>,
    pub(crate) singular_values: HostBuffer<f32>,
}

impl SvdHandle<HostDevice> for HostSvd {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }
    fn u(&self) -> &HostBuffer<f32> {
        &self.u
    }
    fn v(&self) -> &HostBuffer<f32> {
        &self.v
    }
    fn singular_values(&self) -> &HostBuffer<f32> {
        &self.singular_values
    }
}
