//! CUDA's implementation of the generic accelerator layer's device-API seam.
//!
//! This is the whole vendor-specific half of every op family built on
//! [`DeviceApi`]: NVRTC-backed kernel caching, the driver's device-address
//! representation, and the driver calling convention. The host-side
//! orchestration those families need lives once in `hephaestus-core`.

use core::ffi::c_void;
use std::sync::Arc;

use bytemuck::Pod;
use hephaestus_core::{AxisScanKey, CudaC, DeviceApi, LaunchGeometry, Result};
use smallvec::SmallVec;

use crate::CudaDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, SafeCachedKernel, cached_kernel, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;

/// Inline capacity covering every launch the accelerator layer plans today
/// (a scan passes two operands); more spills to the heap rather than failing.
const INLINE_OPERANDS: usize = 4;

/// An opaque handle to a kernel compiled and cached for this device.
///
/// The seam's associated types are public, so the driver module and pipeline
/// key stay crate-private behind these wrappers rather than becoming API.
#[derive(Clone)]
pub struct CudaKernel(Arc<SafeCachedKernel>);

/// An opaque pipeline-cache identity for one compiled kernel specialization.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaKernelCacheKey(PipelineKey);

impl From<AxisScanKey> for CudaKernelCacheKey {
    #[inline]
    fn from(key: AxisScanKey) -> Self {
        Self(PipelineKey::AxisScan {
            marker: key.marker,
            scalar: key.scalar,
            direction: key.direction,
            axis: key.axis,
            width: key.width,
        })
    }
}

impl DeviceApi for CudaDevice {
    type Dialect = CudaC;
    type Kernel = CudaKernel;
    // `CUdeviceptr` is an opaque integer device address.
    type DevicePtr = u64;
    type CacheKey = CudaKernelCacheKey;

    #[inline]
    fn compile_cached(
        &self,
        key: Self::CacheKey,
        entry: &str,
        source: impl FnOnce() -> String,
    ) -> Result<Self::Kernel> {
        cached_kernel(self, key.0, entry, source).map(CudaKernel)
    }

    #[inline]
    fn device_ptr<T: Pod>(buffer: &CudaBuffer<T>) -> Self::DevicePtr {
        buffer.raw()
    }

    #[inline]
    fn buffers_alias<T: Pod>(lhs: &CudaBuffer<T>, rhs: &CudaBuffer<T>) -> bool {
        lhs.aliases(rhs)
    }

    fn launch<P: Pod>(
        &self,
        kernel: &Self::Kernel,
        geometry: LaunchGeometry,
        params: &P,
        operands: &[Self::DevicePtr],
    ) -> Result<()> {
        // The driver reads each argument through a pointer to the *slot*
        // holding it, so the parameter block and every operand address need a
        // local mutable home that outlives the launch call.
        let mut params_slot = *params;
        let mut operand_slots: SmallVec<[Self::DevicePtr; INLINE_OPERANDS]> =
            SmallVec::from_slice(operands);

        let mut args: SmallVec<[*mut c_void; INLINE_OPERANDS + 1]> =
            SmallVec::with_capacity(operand_slots.len() + 1);
        args.push(core::ptr::from_mut(&mut params_slot).cast());
        for slot in &mut operand_slots {
            args.push(core::ptr::from_mut(slot).cast());
        }

        launch_kernel(
            self,
            &kernel.0,
            LaunchConfig::linear_shared(geometry.groups, geometry.width, geometry.shared_bytes),
            &mut args,
        )
    }
}
