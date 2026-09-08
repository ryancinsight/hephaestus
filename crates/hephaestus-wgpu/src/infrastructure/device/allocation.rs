use futures::FutureExt;
use hephaestus_core::{HephaestusError, Result};

use super::WgpuDevice;

impl WgpuDevice {
    pub(super) fn storage_size<T>(&self, len: usize) -> Result<u64> {
        let size = Self::padded_size::<T>(len)?;
        self.validate_buffer_size(size, "WGPU storage")?;
        Ok(size)
    }

    fn validate_buffer_size(&self, size: u64, label: &str) -> Result<()> {
        let maximum = self.device.limits().max_buffer_size;
        if size > maximum {
            return Err(HephaestusError::AllocationFailed {
                message: format!(
                    "{label} requires {size} padded bytes; enabled max_buffer_size={maximum}"
                ),
            });
        }
        Ok(())
    }

    pub(super) fn allocate_buffer(
        &self,
        descriptor: &wgpu::BufferDescriptor<'_>,
    ) -> Result<wgpu::Buffer> {
        let label = descriptor.label.unwrap_or("WGPU buffer");
        self.validate_buffer_size(descriptor.size, label)?;
        self.buffer_operation(label, || Ok(self.device.create_buffer(descriptor)))
    }

    pub(super) fn allocate_storage(
        &self,
        label: &'static str,
        size: u64,
        contents: Option<&[u8]>,
    ) -> Result<wgpu::Buffer> {
        let buffer = self.allocate_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: contents.is_some_and(|bytes| !bytes.is_empty()),
        })?;
        if let Some(bytes) = contents.filter(|bytes| !bytes.is_empty()) {
            self.buffer_operation(label, || {
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

    pub(super) fn buffer_operation<T>(
        &self,
        label: &str,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let out_of_memory = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = operation();
        // Locked wgpu 30's native backend returns ready(scope.error). Poll
        // once rather than blocking a synchronous storage API on a changed
        // provider contract. Pop every scope in reverse order before returning.
        // Native scope stacks and error routing are keyed by the current
        // thread; this synchronous closure never migrates between threads.
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
                        "{label} violated the synchronous native buffer error-scope contract"
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
    fn buffer_error_scopes_keep_concurrent_failures_on_their_threads() -> Result<()> {
        let device = match WgpuDevice::try_default("concurrent buffer scopes") {
            Ok(device) => device,
            Err(HephaestusError::AdapterUnavailable { .. })
                if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_none() =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let tasks: Vec<_> = ["first invalid buffer", "second invalid buffer"]
                .into_iter()
                .map(|label| {
                    let device = &device;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        let outcome = device.buffer_operation(label, || {
                            // Both threads hold all three scopes before either
                            // triggers a real WGPU descriptor validation error.
                            barrier.wait();
                            Ok(device.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some(label),
                                size: 4,
                                usage: wgpu::BufferUsages::empty(),
                                mapped_at_creation: false,
                            }))
                        });
                        match outcome {
                            Err(HephaestusError::DispatchFailed { message }) => {
                                assert!(message.contains(label), "{message}");
                                assert!(message.contains("usage"), "{message}");
                                let other = if label.starts_with("first") {
                                    "second invalid buffer"
                                } else {
                                    "first invalid buffer"
                                };
                                assert!(!message.contains(other), "{message}");
                            }
                            other => {
                                panic!("expected thread-local validation error, got {other:?}")
                            }
                        }
                    })
                })
                .collect();
            for task in tasks {
                task.join()
                    .expect("invariant: each worker completes its error-scope assertions");
            }
        });
        let expected = [7_u32, 11];
        let buffer = device.upload(&expected)?;
        assert_eq!(device.download_owned(&buffer)?, expected);
        Ok(())
    }

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
        let result = device.buffer_operation("invalid allocation", || {
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
