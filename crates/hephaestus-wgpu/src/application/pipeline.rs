//! Shared WGPU pipeline and dispatch utilities.

use std::{hash::Hash, sync::Arc};

use hephaestus_core::{BlockWidth, HephaestusError, Result};
use moirai_gpu::KernelResourceBudget;

use crate::infrastructure::device::{FusionPipelineKey, PipelineKey, WgpuDevice};

/// Encode one bind-and-dispatch compute pass into a caller-owned command stream.
pub(crate) fn encode_compute_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    groups: u32,
    label: &'static str,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}

/// Fetch the cached pipeline for `key`, compiling `source` on first use.
#[must_use]
pub(crate) fn cached_pipeline(
    device: &WgpuDevice,
    key: PipelineKey,
    label: &'static str,
    source: impl FnOnce() -> String,
) -> wgpu::ComputePipeline {
    let cell = device
        .pipeline_cache
        .get_or_insert_with(key, || std::sync::Arc::new(std::sync::OnceLock::new()))
        .expect("invariant: pipeline cache is not poisoned");

    cell.get_or_init(|| {
        let module = device
            .inner()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source().into()),
            });
        device
            .inner()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
    })
    .clone()
}

/// Fetch a cached pipeline while surfacing first-compilation validation.
///
/// WGPU reports shader and pipeline validation through error scopes rather
/// than the constructors' return values. This variant resolves that scope
/// before caching, so an invalid generated kernel becomes a typed preparation
/// failure and can never reach submission.
pub(crate) fn try_cached_pipeline(
    device: &WgpuDevice,
    key: PipelineKey,
    label: &'static str,
    source: impl FnOnce() -> String,
) -> Result<wgpu::ComputePipeline> {
    try_cached_pipeline_in(device, &device.pipeline_cache, key, label, source)
}

fn try_cached_pipeline_in<K>(
    device: &WgpuDevice,
    cache: &moirai_sync::sync::ConcurrentHashMap<
        K,
        Arc<std::sync::OnceLock<wgpu::ComputePipeline>>,
    >,
    key: K,
    label: &'static str,
    source: impl FnOnce() -> String,
) -> Result<wgpu::ComputePipeline>
where
    K: Hash + Eq,
{
    let cell = cache
        .get_or_insert_with(key, || std::sync::Arc::new(std::sync::OnceLock::new()))
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("pipeline cache rejected {label}: {error:?}"),
        })?;
    if let Some(pipeline) = cell.get() {
        return Ok(pipeline.clone());
    }

    let pipeline = compile_pipeline(device, label, source)?;

    // A concurrent preparer may have populated the cell while this pipeline
    // compiled. In either case return the canonical cached instance.
    match cell.set(pipeline) {
        Ok(()) => {}
        Err(concurrent_pipeline) => drop(concurrent_pipeline),
    }
    Ok(cell
        .get()
        .expect("invariant: successful or raced OnceLock initialization stores a pipeline")
        .clone())
}

fn compile_pipeline(
    device: &WgpuDevice,
    label: &'static str,
    source: impl FnOnce() -> String,
) -> Result<wgpu::ComputePipeline> {
    let error_scope = device
        .inner()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device
        .inner()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
    let pipeline = device
        .inner()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
    if let Some(error) = moirai::block_on(error_scope.pop()) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("{label} compilation failed: {error}"),
        });
    }
    Ok(pipeline)
}

fn cached_compilation<T, F>(
    cell: &std::sync::OnceLock<std::result::Result<T, Arc<str>>>,
    compile: F,
) -> Result<T>
where
    T: Clone,
    F: FnOnce() -> Result<T>,
{
    match cell.get_or_init(|| compile().map_err(|error| Arc::<str>::from(error.to_string()))) {
        Ok(value) => Ok(value.clone()),
        Err(message) => Err(HephaestusError::DispatchFailed {
            message: message.to_string(),
        }),
    }
}

/// Fetch a collision-safe cached pipeline for a runtime-generated fusion
/// source while surfacing first-compilation validation.
///
/// All callers for one key share one initialization attempt. The successful
/// pipeline and validation failure are both memoized, so a first-use race
/// cannot duplicate compilation or retry a rejected source.
pub(crate) fn try_cached_fusion_pipeline(
    device: &WgpuDevice,
    key: FusionPipelineKey,
    label: &'static str,
    source: impl FnOnce() -> String,
) -> Result<wgpu::ComputePipeline> {
    let cell = device
        .fusion_pipeline_cache
        .get_or_insert_with(key, || std::sync::Arc::new(std::sync::OnceLock::new()))
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("pipeline cache rejected {label}: {error:?}"),
        })?;
    cached_compilation(&cell, || compile_pipeline(device, label, source))
}

/// Convert a logical work-item count into WGPU workgroup count.
pub(crate) fn workgroups(len: usize, width: BlockWidth) -> Result<u32> {
    let len = u64::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("dispatch size {len} exceeds u64 range"),
    })?;
    let checked =
        width
            .checked_covering_blocks(len)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!("dispatch size {len} exceeds u32 workgroup range"),
            })?;
    let budget = KernelResourceBudget::new(0, 0, width.get())
        .expect("invariant: BlockWidth is non-zero, so budget threads are non-zero");
    let planned = moirai_gpu::plan_launch(budget, len);
    debug_assert_eq!(planned.threads_per_block, width.get());
    debug_assert_eq!(planned.grid_blocks, checked);
    Ok(planned.grid_blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn module_cases_share_process_state() {
        crate::test_support::run_cases(&[
            (
                "workgroups_accepts_exact_u32_group_limit",
                workgroups_accepts_exact_u32_group_limit as fn(),
            ),
            (
                "workgroups_rejects_beyond_u32_group_limit",
                workgroups_rejects_beyond_u32_group_limit as fn(),
            ),
            (
                "cached_compilation_initializes_once_under_concurrency",
                cached_compilation_initializes_once_under_concurrency as fn(),
            ),
            (
                "cached_compilation_replays_failure",
                cached_compilation_replays_failure as fn(),
            ),
        ]);
    }

    fn cached_compilation_initializes_once_under_concurrency() {
        const CALLERS: usize = 8;
        let cell = Arc::new(OnceLock::new());
        let compilations = Arc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(CALLERS);
            for _ in 0..CALLERS {
                let cell = Arc::clone(&cell);
                let compilations = Arc::clone(&compilations);
                handles.push(scope.spawn(move || {
                    cached_compilation(&cell, || {
                        compilations.fetch_add(1, Ordering::Relaxed);
                        std::thread::yield_now();
                        Ok(17_u32)
                    })
                    .expect("cached compilation succeeds")
                }));
            }
            for handle in handles {
                assert_eq!(handle.join().expect("compilation thread completes"), 17);
            }
        });

        assert_eq!(compilations.load(Ordering::Relaxed), 1);
    }

    fn cached_compilation_replays_failure() {
        let cell: OnceLock<std::result::Result<u32, Arc<str>>> = OnceLock::new();
        let compilations = AtomicUsize::new(0);
        for _ in 0..2 {
            match cached_compilation(&cell, || {
                compilations.fetch_add(1, Ordering::Relaxed);
                Err(HephaestusError::DispatchFailed {
                    message: "shader rejected".to_owned(),
                })
            }) {
                Err(HephaestusError::DispatchFailed { message }) => {
                    assert_eq!(message, "kernel dispatch failed: shader rejected")
                }
                other => panic!("expected cached dispatch failure, got {other:?}"),
            }
        }

        assert_eq!(compilations.load(Ordering::Relaxed), 1);
    }

    fn workgroups_accepts_exact_u32_group_limit() {
        let width = BlockWidth::new(256).expect("invariant: test width is non-zero");
        let len: usize = (u64::from(width.get()) * u64::from(u32::MAX))
            .try_into()
            .expect("invariant: max-workgroup test value fits usize on 64-bit");
        match workgroups(len, width) {
            Ok(groups) => assert_eq!(groups, u32::MAX),
            Err(error) => panic!("expected max workgroup count, got {error:?}"),
        }
    }

    fn workgroups_rejects_beyond_u32_group_limit() {
        let width = BlockWidth::new(256).expect("invariant: test width is non-zero");
        let len_u64 = u64::from(width.get()) * u64::from(u32::MAX) + 1;
        let len: usize = len_u64
            .try_into()
            .expect("invariant: overflow test value fits usize on 64-bit");
        match workgroups(len, width) {
            Err(HephaestusError::DispatchFailed { message }) => assert_eq!(
                message,
                format!("dispatch size {len_u64} exceeds u32 workgroup range")
            ),
            other => panic!("expected dispatch-size rejection, got {other:?}"),
        }
    }
}
