//! Resources resolved during preparation and consumed at dispatch.

use bytemuck::Pod;

use crate::domain::accelerator::device_api::{DeviceApi, LaunchGeometry};

use super::metadata;

pub(super) struct WindowLaunch<D: DeviceApi> {
    pub(super) kernel: D::Kernel,
    pub(super) metadata: metadata::WindowMeta,
    pub(super) geometry: LaunchGeometry,
}

/// Prepared generic pooling-forward resources.
pub struct PreparedPoolingForward<'a, D: DeviceApi, T: Pod> {
    pub(super) input: &'a D::Buffer<T>,
    pub(super) output: &'a D::Buffer<T>,
    pub(super) launch: Option<WindowLaunch<D>>,
}

/// Prepared generic pooling-backward resources.
pub struct PreparedPoolingBackward<'a, D: DeviceApi, T: Pod> {
    pub(super) input: Option<&'a D::Buffer<T>>,
    pub(super) grad_output: &'a D::Buffer<T>,
    pub(super) grad_input: &'a D::Buffer<T>,
    pub(super) launch: Option<WindowLaunch<D>>,
}

/// Prepared generic unfold resources.
pub struct PreparedUnfold<'a, D: DeviceApi, T: Pod> {
    pub(super) input: &'a D::Buffer<T>,
    pub(super) output: &'a D::Buffer<T>,
    pub(super) launch: Option<WindowLaunch<D>>,
}

/// Prepared generic fold resources.
pub struct PreparedFold<'a, D: DeviceApi, T: Pod> {
    pub(super) input: &'a D::Buffer<T>,
    pub(super) output: &'a D::Buffer<T>,
    pub(super) launch: Option<WindowLaunch<D>>,
}
