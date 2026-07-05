//! CUDA implementation of the backend-neutral authored-kernel command stream.

use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;

use bytemuck::Pod;
use hephaestus_core::{
    validate_bindings, validate_grouped_bindings, Binding, CommandStream, CudaC, DispatchGrid,
    GroupedBinding, GroupedCommandStream, GroupedKernelDevice, GroupedKernelSequence,
    GroupedKernelSource, HephaestusError, KernelDevice, KernelSource, Result,
};

#[cfg(not(feature = "cuda"))]
use crate::application::pipeline::SafeCachedKernel;
use crate::application::pipeline::{cached_kernel, launch_kernel, LaunchConfig};
use crate::infrastructure::buffer::CudaBuffer;
#[cfg(feature = "cuda")]
use crate::infrastructure::compiler::SafeCachedKernel;
#[cfg(feature = "cuda")]
use crate::infrastructure::device::cuda_byte_count;
use crate::infrastructure::device::CudaDevice;

#[cfg(feature = "cuda")]
type DevicePtr = cuda_oxide::sys::CUdeviceptr;
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

impl<K> core::fmt::Debug for CudaGroupedPrepared<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaGroupedPrepared")
            .field("source_hash", &self.source_hash)
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

impl<K> Clone for CudaGroupedPrepared<K> {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
            source_hash: self.source_hash,
            label: self.label,
            marker: PhantomData,
        }
    }
}

impl<K> core::fmt::Debug for CudaPrepared<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaPrepared")
            .field("source_hash", &self.source_hash)
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

impl<K> Clone for CudaPrepared<K> {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
            source_hash: self.source_hash,
            label: self.label,
            marker: PhantomData,
        }
    }
}

/// CUDA command stream for ordered authored-kernel dispatch, copies, and fills.
///
/// CUDA launches and driver copies are issued to the legacy default stream. The
/// stream ordering contract is therefore enforced by CUDA's default-stream
/// sequencing: operations issued by one `CudaCommandStream` execute in call
/// order, while completion is observed through [`ComputeDevice::synchronize`](hephaestus_core::ComputeDevice::synchronize).
pub struct CudaCommandStream<'d> {
    device: &'d CudaDevice,
}

/// Active CUDA grouped-kernel sequence encoded as ordered launches.
pub struct CudaGroupedSequence<'s> {
    device: &'s CudaDevice,
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
        let cache_key = format!(
            "hephaestus-cuda-stream:{}:{}:{source_hash:016x}",
            K::LABEL,
            K::ENTRY
        );
        let compiled = cached_kernel(self, cache_key, K::ENTRY, || source)?;
        Ok(CudaPrepared {
            kernel: compiled,
            source_hash,
            label: K::LABEL,
            marker: PhantomData,
        })
    }

    fn stream(&self) -> Result<Self::Stream<'_>> {
        self.bind()?;
        Ok(CudaCommandStream { device: self })
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
        let cache_key = format!(
            "hephaestus-cuda-grouped-stream:{}:{}:{source_hash:016x}",
            K::LABEL,
            K::ENTRY
        );
        let compiled = cached_kernel(self, cache_key, K::ENTRY, || source)?;
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

        let mut device_ptrs: Vec<DevicePtr> = bindings.iter().map(|bound| bound.handle).collect();
        let mut params_value = *params;
        let mut args = Vec::with_capacity(device_ptrs.len() + 1);
        args.extend(
            device_ptrs
                .iter_mut()
                .map(|ptr| ptr as *mut DevicePtr as *mut core::ffi::c_void),
        );
        args.push(&mut params_value as *mut K::Params as *mut core::ffi::c_void);

        launch_kernel(
            self.device,
            &prepared.kernel,
            LaunchConfig {
                grid: (grid.x, grid.y, grid.z),
                block: (K::WORKGROUP[0], K::WORKGROUP[1], K::WORKGROUP[2]),
                shared_bytes: K::SHARED_BYTES,
            },
            &mut args,
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
            let byte_count = cuda_byte_count(byte_len, "command stream copy byte count")?;
            // SAFETY: `src` and `dst` are device pointers allocated by this
            // device, lengths are equal, and `byte_len` was derived from that
            // checked element count.
            let res = unsafe { cuda_oxide::sys::cuMemcpyDtoD_v2(dst.raw(), src.raw(), byte_count) };
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

    fn fill_zero<T: Pod>(&mut self, dst: &CudaBuffer<T>) -> Result<()> {
        use hephaestus_core::DeviceBuffer;
        let byte_len = byte_len::<T>(dst.len())?;
        if byte_len == 0 {
            return Ok(());
        }
        self.device.bind()?;
        #[cfg(feature = "cuda")]
        {
            let byte_count = cuda_byte_count(byte_len, "command stream fill byte count")?;
            // SAFETY: `dst` is a device pointer allocated by this device and
            // `byte_len` is the valid allocation byte length for the buffer.
            let res = unsafe { cuda_oxide::sys::cuMemsetD8_v2(dst.raw(), 0, byte_count) };
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

        launch_grouped(self.device, prepared, bindings, params, grid)
    }

    fn encode_grouped_sequence<F>(&mut self, _label: &str, encode: F) -> Result<()>
    where
        F: FnOnce(&mut Self::Sequence<'_>) -> Result<()>,
    {
        let mut sequence = CudaGroupedSequence {
            device: self.device,
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
        launch_grouped(self.device, prepared, bindings, params, grid)
    }
}

fn launch_grouped<K: GroupedKernelSource<CudaC>>(
    device: &CudaDevice,
    prepared: &CudaGroupedPrepared<K>,
    bindings: &[GroupedBinding<'_, CudaDevice>],
    params: &K::Params,
    grid: DispatchGrid,
) -> Result<()> {
    let mut device_ptrs: Vec<DevicePtr> = bindings.iter().map(|bound| bound.handle).collect();
    let mut params_value = *params;
    let mut args = Vec::with_capacity(device_ptrs.len() + 1);
    args.extend(
        device_ptrs
            .iter_mut()
            .map(|ptr| ptr as *mut DevicePtr as *mut core::ffi::c_void),
    );
    args.push(&mut params_value as *mut K::Params as *mut core::ffi::c_void);

    launch_kernel(
        device,
        &prepared.kernel,
        LaunchConfig {
            grid: (grid.x, grid.y, grid.z),
            block: (K::WORKGROUP[0], K::WORKGROUP[1], K::WORKGROUP[2]),
            shared_bytes: K::SHARED_BYTES,
        },
        &mut args,
    )
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
    use hephaestus_core::{
        BindingDecl, ComputeDevice, GroupedBindingDecl, GroupedKernelInterface, KernelInterface,
    };
    use std::borrow::Cow;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ScaleParams {
        len: u32,
        factor: f32,
    }

    #[derive(Clone, Copy, Debug)]
    struct ScaleKernel;

    impl KernelInterface for ScaleKernel {
        type Params = ScaleParams;

        const LABEL: &'static str = "hephaestus-cuda-stream-scale";
        const BINDINGS: &'static [BindingDecl] = &[
            BindingDecl::read_only::<f32>(),
            BindingDecl::read_write::<f32>(),
        ];
        const WORKGROUP: [u32; 3] = [64, 1, 1];
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GroupedParams {
        len: u32,
        addend: f32,
    }

    #[derive(Clone, Copy, Debug)]
    struct GroupedAddKernel;

    impl GroupedKernelInterface for GroupedAddKernel {
        type Params = GroupedParams;

        const LABEL: &'static str = "hephaestus-cuda-grouped-add";
        const BINDINGS: &'static [GroupedBindingDecl] = &[
            GroupedBindingDecl::read_only::<f32>(0, 0),
            GroupedBindingDecl::read_only::<f32>(1, 0),
            GroupedBindingDecl::read_write::<f32>(1, 1),
        ];
        const PARAM_GROUP: u32 = 0;
        const PARAM_BINDING: u32 = 1;
        const WORKGROUP: [u32; 3] = [64, 1, 1];
    }

    impl GroupedKernelSource<CudaC> for GroupedAddKernel {
        const ENTRY: &'static str = "grouped_add_kernel";

        fn source(&self) -> Cow<'static, str> {
            Cow::Borrowed(
                r#"
struct GroupedParams {
    unsigned int len;
    float addend;
};

extern "C" __global__ void grouped_add_kernel(
    const float* left,
    const float* right,
    float* output,
    GroupedParams params
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < params.len) {
        output[idx] = left[idx] + right[idx] + params.addend;
    }
}
"#,
            )
        }
    }

    impl KernelSource<CudaC> for ScaleKernel {
        const ENTRY: &'static str = "scale_kernel";

        fn source(&self) -> Cow<'static, str> {
            Cow::Borrowed(
                r#"
struct ScaleParams {
    unsigned int len;
    float factor;
};

extern "C" __global__ void scale_kernel(
    const float* input,
    float* output,
    ScaleParams params
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < params.len) {
        output[idx] = input[idx] * params.factor;
    }
}
"#,
            )
        }
    }

    #[test]
    fn cuda_command_stream_dispatches_prepared_kernel_when_available() {
        let Ok(device) = CudaDevice::try_default() else {
            eprintln!("CUDA device unavailable, skipping command stream test");
            return;
        };

        let input = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let output = device.alloc_zeroed::<f32>(4).unwrap();
        let prepared = device.prepare(&ScaleKernel).unwrap();
        let bindings = [Binding::read(&input), Binding::read_write(&output)];
        let mut stream = device.stream().unwrap();
        stream
            .encode(
                &prepared,
                &bindings,
                &ScaleParams {
                    len: 4,
                    factor: 2.5,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap();
        stream.submit().unwrap();
        device.synchronize().unwrap();

        let mut out = [0.0_f32; 4];
        device.download(&output, &mut out).unwrap();
        assert_eq!(out, [2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn cuda_command_stream_preserves_fill_copy_dispatch_order_when_available() {
        let Ok(device) = CudaDevice::try_default() else {
            eprintln!("CUDA device unavailable, skipping command stream order test");
            return;
        };

        let input = device.upload(&[2.0_f32, 4.0, 6.0, 8.0]).unwrap();
        let scratch = device.upload(&[9.0_f32, 9.0, 9.0, 9.0]).unwrap();
        let output = device.alloc_zeroed::<f32>(4).unwrap();
        let prepared = device.prepare(&ScaleKernel).unwrap();
        let mut stream = device.stream().unwrap();
        stream.fill_zero(&scratch).unwrap();
        stream.copy(&input, &scratch).unwrap();
        let bindings = [Binding::read(&scratch), Binding::read_write(&output)];
        stream
            .encode(
                &prepared,
                &bindings,
                &ScaleParams {
                    len: 4,
                    factor: 3.0,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap();
        stream.submit().unwrap();
        device.synchronize().unwrap();

        let mut out = [0.0_f32; 4];
        device.download(&output, &mut out).unwrap();
        assert_eq!(out, [6.0, 12.0, 18.0, 24.0]);
    }

    #[test]
    fn cuda_command_stream_rejects_binding_contract_mismatch_when_available() {
        let Ok(device) = CudaDevice::try_default() else {
            eprintln!("CUDA device unavailable, skipping binding mismatch test");
            return;
        };

        let input = device.upload(&[1.0_f32]).unwrap();
        let output = device.alloc_zeroed::<f32>(1).unwrap();
        let prepared = device.prepare(&ScaleKernel).unwrap();
        let bindings = [Binding::read_write(&input), Binding::read_write(&output)];
        let mut stream = device.stream().unwrap();
        let err = stream
            .encode(
                &prepared,
                &bindings,
                &ScaleParams {
                    len: 1,
                    factor: 1.0,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap_err();
        assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
    }

    #[test]
    fn cuda_grouped_command_stream_dispatches_kernel_when_available() {
        let Ok(device) = CudaDevice::try_default() else {
            eprintln!("CUDA device unavailable, skipping grouped command stream test");
            return;
        };

        let left = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let right = device.upload(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
        let output = device.alloc_zeroed::<f32>(4).unwrap();
        let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
        let bindings = [
            GroupedBinding::read(0, 0, &left),
            GroupedBinding::read(1, 0, &right),
            GroupedBinding::read_write(1, 1, &output),
        ];
        let mut stream = device.grouped_stream().unwrap();
        stream
            .encode_grouped(
                &prepared,
                &bindings,
                &GroupedParams {
                    len: 4,
                    addend: 0.5,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap();
        stream.submit_grouped().unwrap();
        device.synchronize().unwrap();

        let mut out = [0.0_f32; 4];
        device.download(&output, &mut out).unwrap();
        assert_eq!(out, [11.5, 22.5, 33.5, 44.5]);
    }

    #[test]
    fn cuda_grouped_command_stream_rejects_group_mismatch_when_available() {
        let Ok(device) = CudaDevice::try_default() else {
            eprintln!("CUDA device unavailable, skipping grouped mismatch test");
            return;
        };

        let left = device.upload(&[1.0_f32]).unwrap();
        let right = device.upload(&[2.0_f32]).unwrap();
        let output = device.alloc_zeroed::<f32>(1).unwrap();
        let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
        let bindings = [
            GroupedBinding::read(0, 0, &left),
            GroupedBinding::read(0, 0, &right),
            GroupedBinding::read_write(1, 1, &output),
        ];
        let mut stream = device.grouped_stream().unwrap();
        let err = stream
            .encode_grouped(
                &prepared,
                &bindings,
                &GroupedParams {
                    len: 1,
                    addend: 0.0,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap_err();
        assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
    }

    #[test]
    fn cuda_grouped_sequence_preserves_order_when_available() {
        let Ok(device) = CudaDevice::try_default() else {
            eprintln!("CUDA device unavailable, skipping grouped sequence test");
            return;
        };

        let left = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let right = device.upload(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
        let scratch = device.alloc_zeroed::<f32>(4).unwrap();
        let output = device.alloc_zeroed::<f32>(4).unwrap();
        let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();

        let mut stream = device.grouped_stream().unwrap();
        stream
            .encode_grouped_sequence("hephaestus-cuda-grouped-sequence", |sequence| {
                sequence.encode_grouped(
                    &prepared,
                    &[
                        GroupedBinding::read(0, 0, &left),
                        GroupedBinding::read(1, 0, &right),
                        GroupedBinding::read_write(1, 1, &scratch),
                    ],
                    &GroupedParams {
                        len: 4,
                        addend: 0.5,
                    },
                    DispatchGrid::new(1, 1, 1),
                )?;
                sequence.encode_grouped(
                    &prepared,
                    &[
                        GroupedBinding::read(0, 0, &scratch),
                        GroupedBinding::read(1, 0, &right),
                        GroupedBinding::read_write(1, 1, &output),
                    ],
                    &GroupedParams {
                        len: 4,
                        addend: 1.0,
                    },
                    DispatchGrid::new(1, 1, 1),
                )
            })
            .unwrap();
        stream.submit_grouped().unwrap();
        device.synchronize().unwrap();

        let mut out = [0.0_f32; 4];
        device.download(&output, &mut out).unwrap();
        assert_eq!(out, [22.5, 43.5, 64.5, 85.5]);
    }
}
