use futures::FutureExt;
use hephaestus_core::{HephaestusError, Result};

use super::WgpuDevice;

impl WgpuDevice {
    pub(super) fn storage_size<T>(&self, len: usize) -> Result<u64> {
        let size = Self::padded_size::<T>(len)?;
        let maximum = self.device.limits().max_buffer_size;
        if size > maximum {
            return Err(HephaestusError::AllocationFailed {
                message: format!(
                    "WGPU storage requires {size} padded bytes; enabled max_buffer_size={maximum}"
                ),
            });
        }
        Ok(size)
    }

    pub(super) fn allocate_storage(
        &self,
        label: &'static str,
        size: u64,
        contents: Option<&[u8]>,
    ) -> Result<wgpu::Buffer> {
        let buffer = self.storage_allocation(label, || {
            Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: contents.is_some_and(|bytes| !bytes.is_empty()),
            }))
        })?;
        if let Some(bytes) = contents.filter(|bytes| !bytes.is_empty()) {
            self.storage_allocation(label, || {
                {
                    let mut mapped = buffer.get_mapped_range_mut(..).map_err(|error| {
                        HephaestusError::AllocationFailed {
                            message: format!("{label} mapped initialization failed: {error}"),
                        }
                    })?;
                    if mapped.len() != bytes.len() {
                        return Err(HephaestusError::AllocationFailed {
                            message: format!(
                                "{label} mapped initialization has {} bytes instead of {}",
                                mapped.len(),
                                bytes.len()
                            ),
                        });
                    }
                    mapped.copy_from_slice(bytes);
                }
                buffer.unmap();
                Ok(())
            })?;
        }
        Ok(buffer)
    }

    fn storage_allocation<T>(
        &self,
        label: &'static str,
        allocate: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let out_of_memory = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = allocate();
        // Locked wgpu 30's native backend returns ready(scope.error). Poll
        // once rather than blocking a synchronous storage API on a changed
        // provider contract. Pop every scope in reverse order before returning.
        let validation = validation.pop().now_or_never();
        let internal = internal.pop().now_or_never();
        let out_of_memory = out_of_memory.pop().now_or_never();
        for outcome in [out_of_memory, internal, validation] {
            if let Some(Some(error)) = outcome {
                return Err(match error {
                    wgpu::Error::OutOfMemory { .. } => HephaestusError::AllocationFailed {
                        message: format!("{label}: {error}"),
                    },
                    wgpu::Error::Validation { .. } | wgpu::Error::Internal { .. } => {
                        HephaestusError::DispatchFailed {
                            message: format!("{label}: {error}"),
                        }
                    }
                });
            }
            if outcome.is_none() {
                return Err(HephaestusError::DispatchFailed {
                    message: format!(
                        "{label} violated the synchronous native allocation error-scope contract"
                    ),
                });
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::WgpuDevice;
    use hephaestus_core::{ComputeDevice, HephaestusError, Result};

    #[test]
    fn allocation_scope_returns_validation_and_releases_nesting() -> Result<()> {
        let device = match WgpuDevice::try_default("allocation error scopes") {
            Ok(device) => device,
            Err(HephaestusError::AdapterUnavailable { .. })
                if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_none() =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let result = device.storage_allocation("invalid allocation", || {
            Ok(device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("invalid allocation"),
                size: 4,
                usage: wgpu::BufferUsages::empty(),
                mapped_at_creation: false,
            }))
        });
        match result {
            Err(HephaestusError::DispatchFailed { message }) => {
                assert!(message.contains("invalid allocation"));
                assert!(message.contains("usage"));
            }
            other => panic!("expected the real WGPU validation error, got {other:?}"),
        }
        let expected = [7u32, 11];
        let uploaded = device.upload(&expected)?;
        assert_eq!(device.download_owned(&uploaded)?, expected);
        Ok(())
    }
}
