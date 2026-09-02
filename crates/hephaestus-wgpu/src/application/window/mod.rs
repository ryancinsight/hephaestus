//! Native WGSL pooling and sliding-window operations.

use bytemuck::Pod;
use hephaestus_core::{DeviceFeature, DialectScalar, HephaestusError, Result, Wgsl};

mod metadata;
mod pooling;
mod prepared;
mod shader;
mod sliding_window;

/// Scalar capability contract shared by WGPU window operation families.
pub trait WgpuWindowScalar: DialectScalar<Wgsl> + Pod + Send + Sync + 'static {
    /// Validate device capabilities required by this scalar representation.
    fn validate_capability(device: &crate::infrastructure::device::WgpuDevice) -> Result<()>;
}

impl WgpuWindowScalar for f32 {
    fn validate_capability(_device: &crate::infrastructure::device::WgpuDevice) -> Result<()> {
        Ok(())
    }
}

impl WgpuWindowScalar for i32 {
    fn validate_capability(_device: &crate::infrastructure::device::WgpuDevice) -> Result<()> {
        Ok(())
    }
}

impl WgpuWindowScalar for u32 {
    fn validate_capability(_device: &crate::infrastructure::device::WgpuDevice) -> Result<()> {
        Ok(())
    }
}

impl WgpuWindowScalar for f64 {
    fn validate_capability(device: &crate::infrastructure::device::WgpuDevice) -> Result<()> {
        if device.supports_device_feature(DeviceFeature::ShaderF64) {
            Ok(())
        } else {
            Err(HephaestusError::InvalidConfiguration {
                message: "WGPU window operations require ShaderF64 for f64".to_owned(),
            })
        }
    }
}

pub use pooling::{PreparedPoolingBackward, PreparedPoolingForward, WgpuPoolingOps};
pub use sliding_window::{
    PreparedSlidingWindowFold, PreparedSlidingWindowUnfold, WgpuSlidingWindowOps,
};

#[cfg(test)]
mod tests;
