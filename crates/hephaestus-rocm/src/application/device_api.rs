//! ROCm's implementation of the generic accelerator layer's device-API seam.
//!
//! This is the whole vendor-specific half of every op family built on
//! [`DeviceApi`](hephaestus_core::DeviceApi): hipRTC-backed kernel caching, HIP's device-address
//! representation, and the HIP calling convention. The host-side
//! orchestration those families need lives once in `hephaestus-core`.

use core::ffi::c_void;
use std::sync::Arc;

use bytemuck::Pod;
use hephaestus_core::{AxisScanKey, DeviceApi, HipC, LaunchGeometry, Result, WindowKey};
use smallvec::SmallVec;

use crate::RocmDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, RocmKernel, cached_kernel, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

/// Inline capacity covering every launch the accelerator layer plans today
/// (a scan passes two operands); more spills to the heap rather than failing.
const INLINE_OPERANDS: usize = 4;

/// An opaque handle to a kernel compiled and cached for this device.
///
/// The seam's associated types are public, so the HIP module and pipeline key
/// stay crate-private behind these wrappers rather than becoming API.
#[derive(Clone)]
pub struct RocmCompiledKernel(Arc<RocmKernel>);

/// An opaque pipeline-cache identity for one compiled kernel specialization.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RocmKernelCacheKey(PipelineKey);

impl From<AxisScanKey> for RocmKernelCacheKey {
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

/// The accelerator layer's spatial-window families — pooling forward and
/// backward, unfold and fold — identify a kernel by operation, host scalar
/// and spatial rank. Without this conversion those families are
/// uninstantiable here, since every one of them is bounded on
/// `D::CacheKey: From<WindowKey>`.
impl From<WindowKey> for RocmKernelCacheKey {
    #[inline]
    fn from(key: WindowKey) -> Self {
        Self(PipelineKey::Window {
            operation: key.operation,
            scalar: key.scalar,
            spatial_rank: key.spatial_rank,
        })
    }
}

impl DeviceApi for RocmDevice {
    type Dialect = HipC;
    type Kernel = RocmCompiledKernel;
    type DevicePtr = DevicePtr;
    type CacheKey = RocmKernelCacheKey;

    #[inline]
    fn compile_cached(
        &self,
        key: Self::CacheKey,
        entry: &str,
        source: impl FnOnce() -> String,
    ) -> Result<Self::Kernel> {
        cached_kernel(self, key.0, entry, source).map(RocmCompiledKernel)
    }

    #[inline]
    fn device_ptr<T: Pod>(buffer: &RocmBuffer<T>) -> Self::DevicePtr {
        buffer.raw()
    }

    #[inline]
    fn buffers_alias<T: Pod>(lhs: &RocmBuffer<T>, rhs: &RocmBuffer<T>) -> bool {
        lhs.aliases(rhs)
    }

    fn launch<P: Pod>(
        &self,
        kernel: &Self::Kernel,
        geometry: LaunchGeometry,
        params: &P,
        operands: &[Self::DevicePtr],
    ) -> Result<()> {
        // HIP reads each argument through a pointer to the *slot* holding it,
        // so the parameter block and every operand address need a local
        // mutable home that outlives the launch call.
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

#[cfg(test)]
mod window_cache_key_tests {
    use super::*;
    use core::any::TypeId;
    use hephaestus_core::WindowOperation;

    // The cache key is deliberately opaque and derives no `Debug`, so these
    // compare with `assert!` and carry the meaning in the message rather than
    // widening a public derive for a test.

    fn key(operation: WindowOperation, spatial_rank: usize) -> WindowKey {
        WindowKey {
            operation,
            scalar: TypeId::of::<f32>(),
            spatial_rank,
        }
    }

    /// The cache is keyed by identity, so two window kernels differing in any
    /// of the three fields must not collide — a collision would hand one
    /// operation's compiled kernel to another.
    #[test]
    fn distinct_windows_convert_to_distinct_cache_keys() {
        let forward = RocmKernelCacheKey::from(key(WindowOperation::PoolingForwardMaximum, 2));
        let average = RocmKernelCacheKey::from(key(WindowOperation::PoolingForwardAverage, 2));
        let rank_three = RocmKernelCacheKey::from(key(WindowOperation::PoolingForwardMaximum, 3));
        let unfold = RocmKernelCacheKey::from(key(WindowOperation::Unfold, 2));
        let double = RocmKernelCacheKey::from(WindowKey {
            operation: WindowOperation::PoolingForwardMaximum,
            scalar: TypeId::of::<f64>(),
            spatial_rank: 2,
        });
        assert!(forward != average, "pooling mode must reach the key");
        assert!(forward != rank_three, "spatial rank must reach the key");
        assert!(forward != unfold, "operation must reach the key");
        assert!(forward != double, "host scalar must reach the key");
    }

    /// The same window converts to the same key, so a second launch of one
    /// kernel hits the cache instead of recompiling.
    #[test]
    fn the_same_window_converts_to_the_same_cache_key() {
        let first = RocmKernelCacheKey::from(key(WindowOperation::Fold, 1));
        let second = RocmKernelCacheKey::from(key(WindowOperation::Fold, 1));
        assert!(
            first == second,
            "the same window must reuse its compiled kernel"
        );
    }
}
