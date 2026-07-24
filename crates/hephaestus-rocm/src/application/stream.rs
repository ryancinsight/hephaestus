//! ROCm implementation of the backend-neutral authored-kernel streams.

use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::rc::Rc;

use bytemuck::Pod;
use hephaestus_core::{
    Binding, CommandStream, DispatchGrid, GroupedBinding, GroupedCommandStream,
    GroupedKernelDevice, GroupedKernelSequence, GroupedKernelSource, HephaestusError, HipC,
    KernelDevice, KernelSource, Result, validate_bindings, validate_grouped_bindings,
};

use crate::RocmDevice;
use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};
use crate::infrastructure::{DevicePtr, RocmBuffer};

/// Prepared ROCm kernel for an authored source type `K`.
pub struct RocmPrepared<K> {
    kernel: Rc<crate::application::pipeline::RocmKernel>,
    source_hash: u64,
    label: &'static str,
    marker: PhantomData<K>,
}

/// Prepared grouped ROCm kernel for an authored source type `K`.
pub struct RocmGroupedPrepared<K> {
    kernel: Rc<crate::application::pipeline::RocmKernel>,
    source_hash: u64,
    label: &'static str,
    marker: PhantomData<K>,
}

impl<K> core::fmt::Debug for RocmPrepared<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RocmPrepared")
            .field("source_hash", &self.source_hash)
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

impl<K> Clone for RocmPrepared<K> {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
            source_hash: self.source_hash,
            label: self.label,
            marker: PhantomData,
        }
    }
}

impl<K> core::fmt::Debug for RocmGroupedPrepared<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RocmGroupedPrepared")
            .field("source_hash", &self.source_hash)
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

impl<K> Clone for RocmGroupedPrepared<K> {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
            source_hash: self.source_hash,
            label: self.label,
            marker: PhantomData,
        }
    }
}

/// ROCm command stream for authored-kernel dispatch, copies, and fills.
///
/// HIP launches and driver operations use the device's default stream. HIP
/// preserves ordering for operations issued to that stream; completion is
/// observed through [`ComputeDevice::synchronize`](hephaestus_core::ComputeDevice::synchronize).
pub struct RocmCommandStream<'d> {
    device: &'d RocmDevice,
}

/// Active grouped ROCm sequence encoded as ordered HIP launches.
pub struct RocmGroupedSequence<'s> {
    device: &'s RocmDevice,
}

impl KernelDevice for RocmDevice {
    type Dialect = HipC;
    type BindingHandle<'a> = DevicePtr;
    type Prepared<K: KernelSource<HipC>> = RocmPrepared<K>;
    type Stream<'d> = RocmCommandStream<'d>;

    #[inline]
    fn binding_handle<T: Pod>(buffer: &Self::Buffer<T>) -> Self::BindingHandle<'_> {
        buffer.raw()
    }

    fn prepare<K: KernelSource<HipC>>(&self, kernel: &K) -> Result<Self::Prepared<K>> {
        let source = kernel.source().into_owned();
        let source_hash = source_hash(K::LABEL, K::ENTRY, &source);
        let compiled = cached_kernel(self, PipelineKey::Stream(source_hash), K::ENTRY, || source)?;
        Ok(RocmPrepared {
            kernel: compiled,
            source_hash,
            label: K::LABEL,
            marker: PhantomData,
        })
    }

    fn stream(&self) -> Result<Self::Stream<'_>> {
        self.bind()?;
        Ok(RocmCommandStream { device: self })
    }
}

impl GroupedKernelDevice for RocmDevice {
    type GroupedPrepared<K: GroupedKernelSource<HipC>> = RocmGroupedPrepared<K>;
    type GroupedStream<'d> = RocmCommandStream<'d>;

    fn prepare_grouped<K: GroupedKernelSource<HipC>>(
        &self,
        kernel: &K,
    ) -> Result<Self::GroupedPrepared<K>> {
        let source = kernel.source().into_owned();
        let source_hash = source_hash(K::LABEL, K::ENTRY, &source);
        let compiled = cached_kernel(
            self,
            PipelineKey::GroupedStream(source_hash),
            K::ENTRY,
            || source,
        )?;
        Ok(RocmGroupedPrepared {
            kernel: compiled,
            source_hash,
            label: K::LABEL,
            marker: PhantomData,
        })
    }

    fn grouped_stream(&self) -> Result<Self::GroupedStream<'_>> {
        self.stream()
    }
}

impl<'d> CommandStream<'d, RocmDevice> for RocmCommandStream<'d> {
    fn encode<K: KernelSource<HipC>>(
        &mut self,
        prepared: &RocmPrepared<K>,
        bindings: &[Binding<'_, RocmDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_bindings::<RocmDevice>(K::LABEL, K::BINDINGS, bindings)?;
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }

        launch_bindings(
            self.device,
            &prepared.kernel,
            K::WORKGROUP,
            K::SHARED_BYTES,
            bindings.iter().map(|binding| binding.handle),
            params,
            grid,
        )
    }

    fn copy<T: Pod>(&mut self, src: &RocmBuffer<T>, dst: &RocmBuffer<T>) -> Result<()> {
        use hephaestus_core::DeviceBuffer;
        if src.len() != dst.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: src.len(),
                device_len: dst.len(),
            });
        }
        copy_device(self.device, src.raw(), dst.raw(), byte_len::<T>(src.len())?)
    }

    fn copy_prefix<T: Pod>(
        &mut self,
        src: &RocmBuffer<T>,
        dst: &RocmBuffer<T>,
        elements: usize,
    ) -> Result<()> {
        use hephaestus_core::DeviceBuffer;
        if elements > src.len() || elements > dst.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: elements,
                device_len: src.len().min(dst.len()),
            });
        }
        copy_device(self.device, src.raw(), dst.raw(), byte_len::<T>(elements)?)
    }

    fn fill_zero<T: Pod>(&mut self, dst: &RocmBuffer<T>) -> Result<()> {
        use hephaestus_core::DeviceBuffer;
        fill_device(self.device, dst.raw(), byte_len::<T>(dst.len())?)
    }

    fn submit(self) -> Result<()> {
        Ok(())
    }
}

impl<'d> GroupedCommandStream<'d, RocmDevice> for RocmCommandStream<'d> {
    type Sequence<'s> = RocmGroupedSequence<'s>;

    fn encode_grouped<K: GroupedKernelSource<HipC>>(
        &mut self,
        prepared: &RocmGroupedPrepared<K>,
        bindings: &[GroupedBinding<'_, RocmDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_grouped_bindings::<RocmDevice>(K::LABEL, K::BINDINGS, bindings)?;
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }

        launch_bindings(
            self.device,
            &prepared.kernel,
            K::WORKGROUP,
            K::SHARED_BYTES,
            bindings.iter().map(|binding| binding.handle),
            params,
            grid,
        )
    }

    fn encode_grouped_sequence<F>(&mut self, _label: &str, encode: F) -> Result<()>
    where
        F: FnOnce(&mut Self::Sequence<'_>) -> Result<()>,
    {
        let mut sequence = RocmGroupedSequence {
            device: self.device,
        };
        encode(&mut sequence)
    }

    fn submit_grouped(self) -> Result<()> {
        CommandStream::submit(self)
    }
}

impl<'s> GroupedKernelSequence<'s, RocmDevice> for RocmGroupedSequence<'s> {
    fn encode_grouped<K: GroupedKernelSource<HipC>>(
        &mut self,
        prepared: &RocmGroupedPrepared<K>,
        bindings: &[GroupedBinding<'_, RocmDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_grouped_bindings::<RocmDevice>(K::LABEL, K::BINDINGS, bindings)?;
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }

        launch_bindings(
            self.device,
            &prepared.kernel,
            K::WORKGROUP,
            K::SHARED_BYTES,
            bindings.iter().map(|binding| binding.handle),
            params,
            grid,
        )
    }
}

fn launch_bindings<P, I>(
    device: &RocmDevice,
    kernel: &crate::application::pipeline::RocmKernel,
    block: [u32; 3],
    shared_bytes: u32,
    bindings: I,
    params: &P,
    grid: DispatchGrid,
) -> Result<()>
where
    P: Pod,
    I: IntoIterator<Item = DevicePtr>,
{
    let mut device_ptrs: Vec<DevicePtr> = bindings.into_iter().collect();
    let mut params_value = *params;
    let mut args = Vec::with_capacity(device_ptrs.len() + 1);
    args.extend(
        device_ptrs
            .iter_mut()
            .map(|ptr| ptr as *mut DevicePtr as *mut core::ffi::c_void),
    );
    args.push(&mut params_value as *mut P as *mut core::ffi::c_void);
    launch_kernel(
        device,
        kernel,
        LaunchConfig {
            grid: (grid.x, grid.y, grid.z),
            block: (block[0], block[1], block[2]),
            shared_bytes,
        },
        &mut args,
    )
}

fn copy_device(device: &RocmDevice, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    device.bind()?;
    #[cfg(all(feature = "rocm", target_os = "linux"))]
    {
        // SAFETY: both pointers are typed allocations owned by this device,
        // and `bytes` was derived from validated buffer lengths.
        let status = unsafe { cubecl_hip_sys::hipMemcpyDtoD(dst, src, bytes) };
        if status == crate::infrastructure::device::HIP_SUCCESS {
            Ok(())
        } else {
            Err(HephaestusError::TransferFailed {
                message: crate::infrastructure::device::status_message(
                    status,
                    "command stream hipMemcpyDtoD",
                ),
            })
        }
    }
    #[cfg(not(all(feature = "rocm", target_os = "linux")))]
    {
        let _ = (src, dst);
        Err(HephaestusError::AdapterUnavailable {
            message:
                "ROCm backend unavailable: enable the `rocm` feature on Linux with ROCm installed"
                    .to_string(),
        })
    }
}

fn fill_device(device: &RocmDevice, dst: DevicePtr, bytes: usize) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    device.bind()?;
    #[cfg(all(feature = "rocm", target_os = "linux"))]
    {
        // SAFETY: `dst` is a typed allocation owned by this device, and
        // `bytes` is its validated allocation size.
        let status = unsafe { cubecl_hip_sys::hipMemsetD8(dst, 0, bytes) };
        if status == crate::infrastructure::device::HIP_SUCCESS {
            Ok(())
        } else {
            Err(HephaestusError::TransferFailed {
                message: crate::infrastructure::device::status_message(
                    status,
                    "command stream hipMemsetD8",
                ),
            })
        }
    }
    #[cfg(not(all(feature = "rocm", target_os = "linux")))]
    {
        let _ = dst;
        Err(HephaestusError::AdapterUnavailable {
            message:
                "ROCm backend unavailable: enable the `rocm` feature on Linux with ROCm installed"
                    .to_string(),
        })
    }
}

fn source_hash(label: &str, entry: &str, source: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    label.hash(&mut hasher);
    entry.hash(&mut hasher);
    source.hash(&mut hasher);
    hasher.finish()
}

fn byte_len<T>(len: usize) -> Result<usize> {
    len.checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| HephaestusError::AllocationFailed {
            message: format!(
                "byte count overflow for {len} elements of size {}",
                core::mem::size_of::<T>()
            ),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_len_reports_checked_element_sizes() {
        assert_eq!(byte_len::<u32>(3).unwrap(), 12);
        assert!(byte_len::<u64>(usize::MAX).is_err());
    }
}
