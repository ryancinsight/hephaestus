#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "ratchet HEPH-UNWRAP-1: pre-existing test-only debt"
    )
)]

//! CUDA implementation of the backend-neutral authored-kernel command stream.

use std::marker::PhantomData;
use std::sync::Arc;

use eunomia::Pod;
use hephaestus_core::{
    Binding, CommandStream, CudaC, DispatchGrid, GroupedBinding, GroupedCommandStream,
    GroupedKernelDevice, GroupedKernelSequence, GroupedKernelSource, HephaestusError, KernelDevice,
    KernelSource, Result, validate_bindings, validate_grouped_bindings,
};

#[cfg(not(feature = "cuda"))]
use crate::application::pipeline::SafeCachedKernel;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, launch_kernel, source_hash,
};
use crate::infrastructure::buffer::CudaBuffer;
#[cfg(feature = "cuda")]
use crate::infrastructure::compiler::SafeCachedKernel;
use crate::infrastructure::device::CudaDevice;

#[cfg(feature = "cuda")]
use crate::infrastructure::buffer::DevicePtr;
#[cfg(not(feature = "cuda"))]
type DevicePtr = u64;

/// Prepared CUDA kernel for a source type `K`.
pub struct CudaPrepared<K> {
    kernel: Arc<SafeCachedKernel>,
    source_hash: u64,
    label: &'static str,
    marker: PhantomData<K>,
}

/// Prepared CUDA kernel for a grouped source type `K`.
pub struct CudaGroupedPrepared<K> {
    kernel: Arc<SafeCachedKernel>,
    source_hash: u64,
    label: &'static str,
    marker: PhantomData<K>,
}

macro_rules! impl_prepared_traits {
    (
        $name:ident {
            debug: [$($debug_field:ident),+ $(,)?],
            clone: [$($clone_field:ident),+ $(,)?]
        }
    ) => {
        impl<K> core::fmt::Debug for $name<K> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let mut debug = f.debug_struct(stringify!($name));
                $(debug.field(stringify!($debug_field), &self.$debug_field);)+
                debug.finish_non_exhaustive()
            }
        }

        impl<K> Clone for $name<K> {
            fn clone(&self) -> Self {
                Self {
                    $($clone_field: self.$clone_field.clone(),)+
                    marker: PhantomData,
                }
            }
        }
    };
}

impl_prepared_traits!(CudaGroupedPrepared {
    debug: [source_hash, label],
    clone: [kernel, source_hash, label]
});
impl_prepared_traits!(CudaPrepared {
    debug: [source_hash, label],
    clone: [kernel, source_hash, label]
});

/// CUDA command stream for ordered authored-kernel dispatch, copies, and fills.
///
/// CUDA launches and driver copies are issued to the legacy default stream. The
/// stream ordering contract is therefore enforced by CUDA's default-stream
/// sequencing: operations issued by one `CudaCommandStream` execute in call
/// order, while completion is observed through [`ComputeDevice::synchronize`](hephaestus_core::ComputeDevice::synchronize).
pub struct CudaCommandStream<'d> {
    device: &'d CudaDevice,
    launch_scratch: CudaLaunchScratch,
}

/// Active CUDA grouped-kernel sequence encoded as ordered launches.
pub struct CudaGroupedSequence<'s> {
    device: &'s CudaDevice,
    launch_scratch: &'s mut CudaLaunchScratch,
}

#[derive(Default)]
struct CudaLaunchScratch {
    device_ptrs: Vec<DevicePtr>,
    args: Vec<*mut core::ffi::c_void>,
}

impl CudaLaunchScratch {
    fn prepare<T, I>(&mut self, handles: I, params: &mut T) -> &mut [*mut core::ffi::c_void]
    where
        I: IntoIterator<Item = DevicePtr>,
    {
        self.device_ptrs.clear();
        self.device_ptrs.extend(handles);
        self.args.clear();
        self.args.reserve(self.device_ptrs.len() + 1);
        self.args.extend(
            self.device_ptrs
                .iter_mut()
                .map(|ptr| ptr as *mut DevicePtr as *mut core::ffi::c_void),
        );
        self.args.push((params as *mut T).cast());
        &mut self.args
    }
}

impl KernelDevice for CudaDevice {
    type Dialect = CudaC;
    type BindingHandle<'a> = DevicePtr;
    type Prepared<K: KernelSource<CudaC>> = CudaPrepared<K>;
    type Stream<'d> = CudaCommandStream<'d>;

    #[inline]
    fn binding_handle<T: Pod>(buffer: &Self::Buffer<T>) -> Self::BindingHandle<'_> {
        buffer.raw()
    }

    fn prepare<K: KernelSource<CudaC>>(&self, kernel: &K) -> Result<Self::Prepared<K>> {
        let source = kernel.source().into_owned();
        let source_hash = source_hash(K::LABEL, K::ENTRY, &source);
        let compiled = cached_kernel(self, PipelineKey::Stream(source_hash), K::ENTRY, || source)?;
        Ok(CudaPrepared {
            kernel: compiled,
            source_hash,
            label: K::LABEL,
            marker: PhantomData,
        })
    }

    fn stream(&self) -> Result<Self::Stream<'_>> {
        self.bind()?;
        Ok(CudaCommandStream {
            device: self,
            launch_scratch: CudaLaunchScratch::default(),
        })
    }
}

impl GroupedKernelDevice for CudaDevice {
    type GroupedPrepared<K: GroupedKernelSource<CudaC>> = CudaGroupedPrepared<K>;
    type GroupedStream<'d> = CudaCommandStream<'d>;

    fn prepare_grouped<K: GroupedKernelSource<CudaC>>(
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
        Ok(CudaGroupedPrepared {
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

impl<'d> CommandStream<'d, CudaDevice> for CudaCommandStream<'d> {
    fn encode<K: KernelSource<CudaC>>(
        &mut self,
        prepared: &CudaPrepared<K>,
        bindings: &[Binding<'_, CudaDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_bindings::<CudaDevice>(K::LABEL, K::BINDINGS, bindings)?;
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }

        let mut params_value = *params;
        let args = self
            .launch_scratch
            .prepare(bindings.iter().map(|bound| bound.handle), &mut params_value);

        launch_kernel(
            self.device,
            &prepared.kernel,
            LaunchConfig {
                grid: (grid.x, grid.y, grid.z),
                block: (K::WORKGROUP[0], K::WORKGROUP[1], K::WORKGROUP[2]),
                shared_bytes: K::SHARED_BYTES,
            },
            args,
        )
    }

    fn copy<T: Pod>(&mut self, src: &CudaBuffer<T>, dst: &CudaBuffer<T>) -> Result<()> {
        use hephaestus_core::DeviceBuffer;
        if src.len() != dst.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: src.len(),
                device_len: dst.len(),
            });
        }
        let byte_len = byte_len::<T>(src.len())?;
        if byte_len == 0 {
            return Ok(());
        }
        self.device.bind()?;
        #[cfg(feature = "cuda")]
        {
            // SAFETY: `src` and `dst` are device pointers allocated by this
            // device, lengths are equal, and `byte_len` was derived from that
            // checked element count.
            let res = unsafe { (self.device.driver().memory.copy)(dst.raw(), src.raw(), byte_len) };
            if res != 0 {
                return Err(HephaestusError::TransferFailed {
                    message: format!(
                        "command stream copy cuMemcpyDtoD_v2({byte_len} bytes) -> {res}"
                    ),
                });
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(HephaestusError::AdapterUnavailable {
                message: "hephaestus-cuda built without the `cuda` feature".to_string(),
            })
        }
    }

    fn copy_prefix<T: Pod>(
        &mut self,
        src: &CudaBuffer<T>,
        dst: &CudaBuffer<T>,
        elements: usize,
    ) -> Result<()> {
        use hephaestus_core::DeviceBuffer;
        if elements > src.len() || elements > dst.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: elements,
                device_len: src.len().min(dst.len()),
            });
        }
        let byte_len = byte_len::<T>(elements)?;
        if byte_len == 0 {
            return Ok(());
        }
        self.device.bind()?;
        #[cfg(feature = "cuda")]
        {
            // SAFETY: `src` and `dst` are device pointers allocated by this
            // device, and `elements` is bounded by both typed buffer lengths.
            let res = unsafe { (self.device.driver().memory.copy)(dst.raw(), src.raw(), byte_len) };
            if res != 0 {
                return Err(HephaestusError::TransferFailed {
                    message: format!(
                        "command stream prefix copy cuMemcpyDtoD_v2({byte_len} bytes) -> {res}"
                    ),
                });
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(HephaestusError::AdapterUnavailable {
                message: "hephaestus-cuda built without the `cuda` feature".to_string(),
            })
        }
    }

    fn fill_zero<T: Pod>(&mut self, dst: &CudaBuffer<T>) -> Result<()> {
        use hephaestus_core::DeviceBuffer;
        let byte_len = byte_len::<T>(dst.len())?;
        if byte_len == 0 {
            return Ok(());
        }
        self.device.bind()?;
        #[cfg(feature = "cuda")]
        {
            // SAFETY: `dst` is a device pointer allocated by this device and
            // `byte_len` is the valid allocation byte length for the buffer.
            let res = unsafe { (self.device.driver().memory.fill)(dst.raw(), 0, byte_len) };
            if res != 0 {
                return Err(HephaestusError::TransferFailed {
                    message: format!(
                        "command stream fill_zero cuMemsetD8_v2({byte_len} bytes) -> {res}"
                    ),
                });
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(HephaestusError::AdapterUnavailable {
                message: "hephaestus-cuda built without the `cuda` feature".to_string(),
            })
        }
    }

    fn submit(self) -> Result<()> {
        Ok(())
    }
}

impl<'d> GroupedCommandStream<'d, CudaDevice> for CudaCommandStream<'d> {
    type Sequence<'s> = CudaGroupedSequence<'s>;

    fn encode_grouped<K: GroupedKernelSource<CudaC>>(
        &mut self,
        prepared: &CudaGroupedPrepared<K>,
        bindings: &[GroupedBinding<'_, CudaDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_grouped_bindings::<CudaDevice>(K::LABEL, K::BINDINGS, bindings)?;
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }

        launch_grouped(
            self.device,
            &mut self.launch_scratch,
            prepared,
            bindings,
            params,
            grid,
        )
    }

    fn encode_grouped_sequence<F>(&mut self, _label: &str, encode: F) -> Result<()>
    where
        F: FnOnce(&mut Self::Sequence<'_>) -> Result<()>,
    {
        let mut sequence = CudaGroupedSequence {
            device: self.device,
            launch_scratch: &mut self.launch_scratch,
        };
        encode(&mut sequence)
    }

    fn submit_grouped(self) -> Result<()> {
        CommandStream::submit(self)
    }
}

impl<'s> GroupedKernelSequence<'s, CudaDevice> for CudaGroupedSequence<'s> {
    fn encode_grouped<K: GroupedKernelSource<CudaC>>(
        &mut self,
        prepared: &CudaGroupedPrepared<K>,
        bindings: &[GroupedBinding<'_, CudaDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_grouped_bindings::<CudaDevice>(K::LABEL, K::BINDINGS, bindings)?;
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }
        launch_grouped(
            self.device,
            self.launch_scratch,
            prepared,
            bindings,
            params,
            grid,
        )
    }
}

fn launch_grouped<K: GroupedKernelSource<CudaC>>(
    device: &CudaDevice,
    launch_scratch: &mut CudaLaunchScratch,
    prepared: &CudaGroupedPrepared<K>,
    bindings: &[GroupedBinding<'_, CudaDevice>],
    params: &K::Params,
    grid: DispatchGrid,
) -> Result<()> {
    let mut params_value = *params;
    let args = launch_scratch.prepare(bindings.iter().map(|bound| bound.handle), &mut params_value);

    launch_kernel(
        device,
        &prepared.kernel,
        LaunchConfig {
            grid: (grid.x, grid.y, grid.z),
            block: (K::WORKGROUP[0], K::WORKGROUP[1], K::WORKGROUP[2]),
            shared_bytes: K::SHARED_BYTES,
        },
        args,
    )
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
mod tests;
