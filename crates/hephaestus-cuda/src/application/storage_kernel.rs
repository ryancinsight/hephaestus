//! CUDA implementation of backend-neutral multi-storage kernels.

use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use bytemuck::Pod;
use hephaestus_core::{
    BinaryStorageKernel, DeviceBuffer, DispatchGrid, HephaestusError, MultiStorageDevice,
    MultiStorageKernel, Result, UnaryStorageKernel,
};

use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

#[cfg(feature = "cuda")]
type DevicePtr = cuda_oxide::sys::CUdeviceptr;
#[cfg(not(feature = "cuda"))]
type DevicePtr = u64;

/// One typed CUDA storage-buffer binding for a multi-storage CUDA C kernel.
#[derive(Clone, Copy, Debug)]
pub struct CudaStorageBinding<'a> {
    binding: u32,
    ptr: DevicePtr,
    marker: PhantomData<&'a ()>,
}

impl<'a> CudaStorageBinding<'a> {
    /// Bind `buffer` to a CUDA storage slot.
    #[must_use]
    #[inline]
    pub fn new<T: Pod>(binding: u32, buffer: &'a CudaBuffer<T>) -> Self {
        Self {
            binding,
            ptr: buffer.raw(),
            marker: PhantomData,
        }
    }
}

impl MultiStorageDevice for CudaDevice {
    type StorageBinding<'a> = CudaStorageBinding<'a>;

    fn storage_binding<T: Pod>(binding: u32, buffer: &Self::Buffer<T>) -> Self::StorageBinding<'_> {
        CudaStorageBinding::new(binding, buffer)
    }
}

/// CUDA C kernel with N storage buffers and one POD parameter block.
#[derive(Debug)]
pub struct CudaMultiStorageKernel {
    source: &'static str,
    entry_point: &'static str,
    storage_bindings: Vec<u32>,
    block: [u32; 3],
    shared_bytes: u32,
    source_hash: u64,
}

impl CudaMultiStorageKernel {
    /// Construct a CUDA C multi-storage-buffer kernel.
    ///
    /// `storage_bindings` declares the expected flat pointer argument order.
    /// The POD parameter block is passed as the final CUDA kernel argument.
    ///
    /// # Errors
    /// Returns [`HephaestusError::DispatchFailed`] when the entry point is
    /// empty, no storage bindings are declared, a binding is duplicated, or a
    /// block dimension is zero.
    pub fn new(
        label: &'static str,
        source: &'static str,
        entry_point: &'static str,
        storage_bindings: &[u32],
        block: [u32; 3],
        shared_bytes: u32,
    ) -> Result<Self> {
        if entry_point.is_empty() {
            return Err(HephaestusError::DispatchFailed {
                message: "CUDA multi-storage kernel entry point is empty".to_string(),
            });
        }
        if storage_bindings.is_empty() {
            return Err(HephaestusError::DispatchFailed {
                message: "CUDA multi-storage kernel has no storage bindings".to_string(),
            });
        }
        if block.contains(&0) {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "CUDA multi-storage kernel block contains zero dimension: {block:?}"
                ),
            });
        }
        validate_distinct_bindings(storage_bindings)?;

        Ok(Self {
            source,
            entry_point,
            storage_bindings: storage_bindings.to_vec(),
            block,
            shared_bytes,
            source_hash: source_hash(label, entry_point, source),
        })
    }
}

impl<'a, P: Pod, const N: usize> MultiStorageKernel<CudaDevice, P, [CudaStorageBinding<'a>; N]>
    for CudaMultiStorageKernel
{
    fn dispatch(
        &self,
        device: &CudaDevice,
        bindings: [CudaStorageBinding<'a>; N],
        params: &P,
        grid: DispatchGrid,
    ) -> Result<()> {
        if N != self.storage_bindings.len() {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "CUDA multi-storage kernel expected {} storage bindings, got {N}",
                    self.storage_bindings.len()
                ),
            });
        }
        for (expected, actual) in self.storage_bindings.iter().zip(bindings.iter()) {
            if *expected != actual.binding {
                return Err(HephaestusError::DispatchFailed {
                    message: format!(
                        "CUDA multi-storage kernel expected binding {expected}, got {}",
                        actual.binding
                    ),
                });
            }
        }
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }

        let compiled = cached_kernel(
            device,
            PipelineKey::MultiStorage(self.source_hash),
            self.entry_point,
            || self.source.to_string(),
        )?;

        let mut device_ptrs: Vec<DevicePtr> = bindings.iter().map(|binding| binding.ptr).collect();
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
            &compiled,
            LaunchConfig {
                grid: (grid.x, grid.y, grid.z),
                block: (self.block[0], self.block[1], self.block[2]),
                shared_bytes: self.shared_bytes,
            },
            &mut args,
        )
    }
}

impl<T, P> UnaryStorageKernel<CudaDevice, T, P> for CudaMultiStorageKernel
where
    T: Pod,
    P: Pod,
{
    fn dispatch(
        &self,
        device: &CudaDevice,
        input: &CudaBuffer<T>,
        output: &CudaBuffer<T>,
        params: &P,
        grid: DispatchGrid,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: input.len(),
                device_len: output.len(),
            });
        }

        MultiStorageKernel::<CudaDevice, P, [CudaStorageBinding<'_>; 2]>::dispatch(
            self,
            device,
            [
                CudaStorageBinding::new(0, input),
                CudaStorageBinding::new(1, output),
            ],
            params,
            grid,
        )
    }
}

impl<T, P> BinaryStorageKernel<CudaDevice, T, P> for CudaMultiStorageKernel
where
    T: Pod,
    P: Pod,
{
    fn dispatch(
        &self,
        device: &CudaDevice,
        left: &CudaBuffer<T>,
        right: &CudaBuffer<T>,
        output: &CudaBuffer<T>,
        params: &P,
        grid: DispatchGrid,
    ) -> Result<()> {
        if left.len() != right.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: left.len(),
                device_len: right.len(),
            });
        }
        if left.len() != output.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: left.len(),
                device_len: output.len(),
            });
        }

        MultiStorageKernel::<CudaDevice, P, [CudaStorageBinding<'_>; 3]>::dispatch(
            self,
            device,
            [
                CudaStorageBinding::new(0, left),
                CudaStorageBinding::new(1, right),
                CudaStorageBinding::new(2, output),
            ],
            params,
            grid,
        )
    }
}

fn validate_distinct_bindings(storage_bindings: &[u32]) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for binding in storage_bindings {
        if !seen.insert(*binding) {
            return Err(HephaestusError::DispatchFailed {
                message: format!("duplicate CUDA storage binding {binding}"),
            });
        }
    }
    Ok(())
}

fn source_hash(label: &str, entry: &str, source: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    label.hash(&mut hasher);
    entry.hash(&mut hasher);
    source.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_storage_layout_rejects_duplicate_storage_binding() {
        let err = CudaMultiStorageKernel::new(
            "duplicate",
            "extern \"C\" __global__ void duplicate(float*, float*, unsigned int) {}",
            "duplicate",
            &[0, 0],
            [8, 8, 1],
            0,
        )
        .unwrap_err();

        assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
    }

    #[test]
    fn multi_storage_layout_rejects_zero_block_dimension() {
        let err = CudaMultiStorageKernel::new(
            "zero_block",
            "extern \"C\" __global__ void zero_block(float*, unsigned int) {}",
            "zero_block",
            &[0],
            [8, 0, 1],
            0,
        )
        .unwrap_err();

        assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
    }
}
