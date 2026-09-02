//! Generic C-family unfold and fold launches.

use core::marker::PhantomData;

use bytemuck::Pod;

use crate::domain::accelerator::device_api::DeviceApi;
use crate::domain::dialect::DialectScalar;
use crate::domain::error::Result;
use crate::domain::sliding_window::{
    SlidingWindowFoldOperands, SlidingWindowOps, SlidingWindowUnfoldOperands,
    plan_sliding_window_fold, plan_sliding_window_unfold,
};

use super::key::{WindowKey, WindowOperation};
use super::launch::{elements, launch_window, prepare_launch, validate_spatial_rank};
use super::metadata;
use super::prepared::{PreparedFold, PreparedUnfold};
use super::source::WindowDialect;

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
