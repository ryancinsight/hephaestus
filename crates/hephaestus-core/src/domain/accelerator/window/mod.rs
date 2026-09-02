//! Generic pooling and sliding-window launches over runtime-compiled C-family devices.
//!
//! CUDA and HIP share one source generator and one host-side implementation.
//! The vendor crates provide only their DeviceApi mechanics and convert
//! WindowKey into their local pipeline key.

mod metadata;
mod source;

use core::any::TypeId;
use core::marker::PhantomData;

use bytemuck::Pod;
use leto::Layout;

use crate::domain::accelerator::device_api::{DeviceApi, LaunchGeometry};
use crate::domain::dialect::DialectScalar;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::pooling::{
    PoolingBackwardOperands, PoolingForwardOperands, PoolingMode, PoolingOps,
    plan_pooling_backward, plan_pooling_forward,
};
use crate::domain::sliding_window::{
    SlidingWindowFoldOperands, SlidingWindowOps, SlidingWindowUnfoldOperands,
    plan_sliding_window_fold, plan_sliding_window_unfold,
};

pub use source::{WindowDialect, c_family_window_source};

/// A generated spatial-window kernel specialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WindowOperation {
    /// Forward maximum pooling.
    PoolingForwardMaximum,
    /// Forward average pooling.
    PoolingForwardAverage,
    /// Maximum-pooling input gradient accumulation.
    PoolingBackwardMaximum,
    /// Average-pooling input gradient accumulation.
    PoolingBackwardAverage,
    /// Extract spatial windows into column storage.
    Unfold,
    /// Accumulate column storage into a spatial tensor.
    Fold,
}

/// Pipeline-cache identity for one generated spatial-window kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WindowKey {
    /// Operation and pooling mode represented by the source.
    pub operation: WindowOperation,
    /// Host scalar represented by the source.
    pub scalar: TypeId,
    /// Number of spatial axes represented by the metadata.
    pub spatial_rank: usize,
}

struct WindowLaunch<D: DeviceApi> {
    kernel: D::Kernel,
    metadata: metadata::WindowMeta,
    geometry: LaunchGeometry,
}

/// Prepared generic pooling-forward resources.
pub struct PreparedPoolingForward<'a, D: DeviceApi, T: Pod> {
    input: &'a D::Buffer<T>,
    output: &'a D::Buffer<T>,
    launch: Option<WindowLaunch<D>>,
}

/// Prepared generic pooling-backward resources.
pub struct PreparedPoolingBackward<'a, D: DeviceApi, T: Pod> {
    input: Option<&'a D::Buffer<T>>,
    grad_output: &'a D::Buffer<T>,
    grad_input: &'a D::Buffer<T>,
    launch: Option<WindowLaunch<D>>,
}

/// Prepared generic unfold resources.
pub struct PreparedUnfold<'a, D: DeviceApi, T: Pod> {
    input: &'a D::Buffer<T>,
    output: &'a D::Buffer<T>,
    launch: Option<WindowLaunch<D>>,
}

/// Prepared generic fold resources.
pub struct PreparedFold<'a, D: DeviceApi, T: Pod> {
    input: &'a D::Buffer<T>,
    output: &'a D::Buffer<T>,
    launch: Option<WindowLaunch<D>>,
}

/// Generic C-family implementation of pooling operations.
#[derive(Clone, Copy, Debug)]
pub struct CFamilyPoolingOps<D>(PhantomData<fn() -> D>);

impl<D> CFamilyPoolingOps<D> {
    /// Construct the pooling operation marker for device D.
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<D> Default for CFamilyPoolingOps<D> {
    fn default() -> Self {
        Self::new()
    }
}

/// Generic C-family implementation of sliding-window operations.
#[derive(Clone, Copy, Debug)]
pub struct CFamilySlidingWindowOps<D>(PhantomData<fn() -> D>);

impl<D> CFamilySlidingWindowOps<D> {
    /// Construct the sliding-window operation marker for device D.
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<D> Default for CFamilySlidingWindowOps<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D, T> PoolingOps<D, T> for CFamilyPoolingOps<D>
where
    D: DeviceApi + 'static,
    D::CacheKey: From<WindowKey>,
    D::Dialect: WindowDialect,
    T: DialectScalar<D::Dialect> + Pod + 'static,
{
    type PreparedForward<'a, const R: usize, const S: usize>
        = PreparedPoolingForward<'a, D, T>
    where
        D: 'a,
        T: 'a;
    type PreparedBackward<'a, const R: usize, const S: usize>
        = PreparedPoolingBackward<'a, D, T>
    where
        D: 'a,
        T: 'a;

    fn prepare_pooling_forward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a D,
        operands: PoolingForwardOperands<'a, D::Buffer<T>, R>,
        parameters: leto::WindowParameters<S>,
        mode: PoolingMode,
    ) -> Result<Self::PreparedForward<'a, R, S>> {
        validate_spatial_rank::<S>()?;
        let plan = plan_pooling_forward::<T, _, R, S>(
            &operands,
            parameters,
            D::buffers_alias(operands.input.buffer, operands.output.buffer),
        )?;
        plan.validate_address_limit(i32::MAX as usize)?;
        let metadata = metadata::WindowMeta::new(
            metadata::WindowLayoutMeta::new(operands.input.layout)?,
            metadata::WindowLayoutMeta::new(operands.output.layout)?,
            metadata::WindowLayoutMeta::empty(),
            &plan.geometry,
        )?;
        let launch = prepare_launch::<D, T>(
            device,
            pooling_forward_operation(mode),
            metadata,
            elements(operands.output.layout, "pooling output")?,
            S,
        )?;
        Ok(PreparedPoolingForward {
            input: operands.input.buffer,
            output: operands.output.buffer,
            launch,
        })
    }

    fn dispatch_pooling_forward<const R: usize, const S: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedForward<'_, R, S>,
    ) -> Result<()> {
        let Some(launch) = &prepared.launch else {
            return Ok(());
        };
        launch_window(
            device,
            launch,
            &[
                D::device_ptr(prepared.input),
                D::device_ptr(prepared.output),
            ],
        )
    }

    fn prepare_pooling_backward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a D,
        operands: PoolingBackwardOperands<'a, D::Buffer<T>, R>,
        parameters: leto::WindowParameters<S>,
        mode: PoolingMode,
    ) -> Result<Self::PreparedBackward<'a, R, S>> {
        validate_spatial_rank::<S>()?;
        let illegal_aliasing = operands
            .input
            .is_some_and(|input| D::buffers_alias(input.buffer, operands.grad_input.buffer))
            || D::buffers_alias(operands.grad_output.buffer, operands.grad_input.buffer);
        let plan =
            plan_pooling_backward::<T, _, R, S>(&operands, parameters, mode, illegal_aliasing)?;
        plan.validate_address_limit(i32::MAX as usize)?;
        let input_layout = operands
            .input
            .map_or(operands.grad_input.layout, |input| input.layout);
        let metadata = metadata::WindowMeta::new(
            metadata::WindowLayoutMeta::new(input_layout)?,
            metadata::WindowLayoutMeta::new(operands.grad_output.layout)?,
            metadata::WindowLayoutMeta::new(operands.grad_input.layout)?,
            &plan.geometry,
        )?;
        let launch = prepare_launch::<D, T>(
            device,
            pooling_backward_operation(mode),
            metadata,
            elements(operands.grad_input.layout, "pooling gradient input")?,
            S,
        )?;
        Ok(PreparedPoolingBackward {
            input: operands.input.map(|input| input.buffer),
            grad_output: operands.grad_output.buffer,
            grad_input: operands.grad_input.buffer,
            launch,
        })
    }

    fn dispatch_pooling_backward<const R: usize, const S: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedBackward<'_, R, S>,
    ) -> Result<()> {
        let Some(launch) = &prepared.launch else {
            return Ok(());
        };
        match prepared.input {
            Some(input) => launch_window(
                device,
                launch,
                &[
                    D::device_ptr(input),
                    D::device_ptr(prepared.grad_output),
                    D::device_ptr(prepared.grad_input),
                ],
            ),
            None => launch_window(
                device,
                launch,
                &[
                    D::device_ptr(prepared.grad_output),
                    D::device_ptr(prepared.grad_input),
                ],
            ),
        }
    }
}

impl<D, T> SlidingWindowOps<D, T> for CFamilySlidingWindowOps<D>
where
    D: DeviceApi + 'static,
    D::CacheKey: From<WindowKey>,
    D::Dialect: WindowDialect,
    T: DialectScalar<D::Dialect> + Pod + 'static,
{
    type PreparedUnfold<'a, const R: usize, const S: usize>
        = PreparedUnfold<'a, D, T>
    where
        D: 'a,
        T: 'a;
    type PreparedFold<'a, const R: usize, const S: usize>
        = PreparedFold<'a, D, T>
    where
        D: 'a,
        T: 'a;

    fn prepare_unfold<'a, const R: usize, const S: usize>(
        &self,
        device: &'a D,
        operands: SlidingWindowUnfoldOperands<'a, D::Buffer<T>, R>,
        parameters: leto::WindowParameters<S>,
    ) -> Result<Self::PreparedUnfold<'a, R, S>> {
        validate_spatial_rank::<S>()?;
        let plan = plan_sliding_window_unfold::<T, _, R, S>(
            &operands,
            parameters,
            D::buffers_alias(operands.input.buffer, operands.output.buffer),
        )?;
        plan.validate_address_limit(i32::MAX as usize)?;
        let metadata = metadata::WindowMeta::new(
            metadata::WindowLayoutMeta::new(operands.input.layout)?,
            metadata::WindowLayoutMeta::new(operands.output.layout)?,
            metadata::WindowLayoutMeta::empty(),
            &plan.geometry,
        )?;
        let launch = prepare_launch::<D, T>(
            device,
            WindowOperation::Unfold,
            metadata,
            elements(operands.output.layout, "unfold output")?,
            S,
        )?;
        Ok(PreparedUnfold {
            input: operands.input.buffer,
            output: operands.output.buffer,
            launch,
        })
    }

    fn dispatch_unfold<const R: usize, const S: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedUnfold<'_, R, S>,
    ) -> Result<()> {
        let Some(launch) = &prepared.launch else {
            return Ok(());
        };
        launch_window(
            device,
            launch,
            &[
                D::device_ptr(prepared.input),
                D::device_ptr(prepared.output),
            ],
        )
    }

    fn prepare_fold<'a, const R: usize, const S: usize>(
        &self,
        device: &'a D,
        operands: SlidingWindowFoldOperands<'a, D::Buffer<T>, R>,
        output_spatial_shape: [usize; S],
        parameters: leto::WindowParameters<S>,
    ) -> Result<Self::PreparedFold<'a, R, S>> {
        validate_spatial_rank::<S>()?;
        let plan = plan_sliding_window_fold::<T, _, R, S>(
            &operands,
            output_spatial_shape,
            parameters,
            D::buffers_alias(operands.input.buffer, operands.output.buffer),
        )?;
        plan.validate_address_limit(i32::MAX as usize)?;
        let metadata = metadata::WindowMeta::new(
            metadata::WindowLayoutMeta::new(operands.input.layout)?,
            metadata::WindowLayoutMeta::empty(),
            metadata::WindowLayoutMeta::new(operands.output.layout)?,
            &plan.geometry,
        )?;
        let launch = prepare_launch::<D, T>(
            device,
            WindowOperation::Fold,
            metadata,
            elements(operands.output.layout, "fold output")?,
            S,
        )?;
        Ok(PreparedFold {
            input: operands.input.buffer,
            output: operands.output.buffer,
            launch,
        })
    }

    fn dispatch_fold<const R: usize, const S: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedFold<'_, R, S>,
    ) -> Result<()> {
        let Some(launch) = &prepared.launch else {
            return Ok(());
        };
        launch_window(
            device,
            launch,
            &[
                D::device_ptr(prepared.input),
                D::device_ptr(prepared.output),
            ],
        )
    }
}

fn pooling_forward_operation(mode: PoolingMode) -> WindowOperation {
    match mode {
        PoolingMode::Maximum => WindowOperation::PoolingForwardMaximum,
        PoolingMode::Average => WindowOperation::PoolingForwardAverage,
    }
}

fn pooling_backward_operation(mode: PoolingMode) -> WindowOperation {
    match mode {
        PoolingMode::Maximum => WindowOperation::PoolingBackwardMaximum,
        PoolingMode::Average => WindowOperation::PoolingBackwardAverage,
    }
}

fn validate_spatial_rank<const S: usize>() -> Result<()> {
    if !(1..=3).contains(&S) {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "C-family window operations support spatial ranks 1 through 3, got {S}"
            ),
        });
    }
    Ok(())
}

fn elements<const R: usize>(layout: &Layout<R>, name: &str) -> Result<usize> {
    layout
        .checked_size()
        .map_err(|error| HephaestusError::InvalidConfiguration {
            message: format!("{name} layout rejected: {error}"),
        })
}

fn prepare_launch<D, T>(
    device: &D,
    operation: WindowOperation,
    metadata: metadata::WindowMeta,
    elements: usize,
    spatial_rank: usize,
) -> Result<Option<WindowLaunch<D>>>
where
    D: DeviceApi,
    D::CacheKey: From<WindowKey>,
    D::Dialect: WindowDialect,
    T: DialectScalar<D::Dialect> + Pod + 'static,
{
    if elements == 0 {
        return Ok(None);
    }
    let width = crate::domain::launch::BlockWidth::DEFAULT;
    let work_items = u64::try_from(elements).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("window element count {elements} exceeds u64 range"),
    })?;
    let groups = width.checked_covering_blocks(work_items).ok_or_else(|| {
        HephaestusError::DispatchFailed {
            message: format!("window element count {elements} exceeds u32 grid range"),
        }
    })?;
    let key = WindowKey {
        operation,
        scalar: TypeId::of::<T>(),
        spatial_rank,
    };
    let kernel = device.compile_cached(D::CacheKey::from(key), source::WINDOW_ENTRY, || {
        <D::Dialect as WindowDialect>::window_source::<T>(operation, width)
    })?;
    Ok(Some(WindowLaunch {
        kernel,
        metadata,
        geometry: LaunchGeometry::linear(groups, width),
    }))
}

fn launch_window<D: DeviceApi>(
    device: &D,
    launch: &WindowLaunch<D>,
    operands: &[D::DevicePtr],
) -> Result<()> {
    device.launch(&launch.kernel, launch.geometry, &launch.metadata, operands)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::dialect::{CudaC, HipC};

    #[test]
    fn cuda_and_hip_window_sources_are_identical() {
        let width = crate::BlockWidth::new(64).expect("non-zero test width");
        for operation in [
            WindowOperation::PoolingForwardMaximum,
            WindowOperation::PoolingForwardAverage,
            WindowOperation::PoolingBackwardMaximum,
            WindowOperation::PoolingBackwardAverage,
            WindowOperation::Unfold,
            WindowOperation::Fold,
        ] {
            assert_eq!(
                c_family_window_source::<CudaC, f32>(operation, width),
                c_family_window_source::<HipC, f32>(operation, width)
            );
        }
    }

    #[test]
    fn source_varies_by_operation_and_scalar() {
        let width = crate::BlockWidth::new(32).expect("non-zero test width");
        let maximum =
            c_family_window_source::<CudaC, f32>(WindowOperation::PoolingForwardMaximum, width);
        let average =
            c_family_window_source::<CudaC, f32>(WindowOperation::PoolingForwardAverage, width);
        let integer =
            c_family_window_source::<CudaC, i32>(WindowOperation::PoolingForwardMaximum, width);
        assert_ne!(maximum, average);
        assert_ne!(maximum, integer);
        assert!(maximum.contains("float"));
        assert!(integer.contains("int"));
    }
}
