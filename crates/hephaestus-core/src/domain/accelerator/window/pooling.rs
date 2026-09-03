//! Generic C-family pooling forward and backward launches.

use core::marker::PhantomData;

use eunomia::Pod;

use crate::domain::accelerator::device_api::DeviceApi;
use crate::domain::dialect::DialectScalar;
use crate::domain::error::Result;
use crate::domain::pooling::{
    PoolingBackwardOperands, PoolingForwardOperands, PoolingMode, PoolingOps,
    plan_pooling_backward, plan_pooling_forward,
};

use super::key::WindowKey;
use super::launch::{
    elements, launch_window, pooling_backward_operation, pooling_forward_operation, prepare_launch,
    validate_spatial_rank,
};
use super::metadata;
use super::prepared::{PreparedPoolingBackward, PreparedPoolingForward};
use super::source::WindowDialect;

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
