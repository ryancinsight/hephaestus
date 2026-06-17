//! Scan operations.

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;
use hephaestus_core::{BlockWidth, Result};
use hephaestus_wgpu as wgpu_backend;

pub use wgpu_backend::{CumProdOp, CumSumOp, ScanDirection};

/// Scan a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn scan_axis<Op, T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'_, T, 2>,
    axis: usize,
    direction: ScanDirection,
    width: BlockWidth,
) -> Result<MetalBuffer<T>>
where
    Op: wgpu_backend::ScanWgslOp,
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ScanIdentity<Op>,
{
    let inner = wgpu_backend::scan_axis::<Op, T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        direction,
        width,
    )?;
    Ok(MetalBuffer { inner })
}

/// Scan a rank-2 strided matrix along `axis`, preserving the input shape.
#[inline]
pub fn scan_axis_into<Op, T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'_, T, 2>,
    axis: usize,
    direction: ScanDirection,
    out: crate::application::strided::StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    Op: wgpu_backend::ScanWgslOp,
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ScanIdentity<Op>,
{
    wgpu_backend::scan_axis_into::<Op, T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        direction,
        crate::application::strided::to_wgpu_strided(out),
        width,
    )
}

/// Forward cumulative sum over a rank-2 strided matrix, allocating a C-contiguous output buffer.
#[inline]
pub fn cumsum<T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<MetalBuffer<T>>
where
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ScanIdentity<CumSumOp>,
{
    let inner = wgpu_backend::cumsum::<T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        width,
    )?;
    Ok(MetalBuffer { inner })
}

/// Forward cumulative sum over a rank-2 strided matrix along `axis`.
#[inline]
pub fn cumsum_into<T>(
    device: &MetalDevice,
    input: crate::application::strided::StridedOperand<'_, T, 2>,
    axis: usize,
    out: crate::application::strided::StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: wgpu_backend::WgslScalar + bytemuck::Pod + wgpu_backend::ScanIdentity<CumSumOp>,
{
    wgpu_backend::cumsum_into::<T>(
        &device.inner,
        crate::application::strided::to_wgpu_strided(input),
        axis,
        crate::application::strided::to_wgpu_strided(out),
        width,
    )
}
