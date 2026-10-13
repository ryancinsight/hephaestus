//! Packed matrix-layout metadata and kernel-cache keys.

use core::marker::PhantomData;
use eunomia::{Pod, Zeroable};

/// Packed layout metadata matching the WGSL `MatrixLayout` uniform.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuMatrixLayout {
    pub(crate) shape: [u32; 2],
    pub(crate) strides: [i32; 2],
    pub(crate) offset: u32,
    pub(crate) _pad: [u32; 3], // pad to 32 bytes (multiple of 16)
}

pub(crate) struct MatmulKernel<T>(PhantomData<T>);

pub(crate) struct KronKernel<T>(PhantomData<T>);

pub(crate) struct IdentityKernel<T>(PhantomData<T>);
