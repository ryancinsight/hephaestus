//! Leto-backed host implementation of the regular and transposed convolution
//! seams (ADR 0046).
//!
//! Unlike [`crate::attention`] and [`crate::cross_entropy`], convolution
//! carries no value-dependent semantic status word: `hephaestus-wgpu`'s
//! `application/convolution` module validates only shape, storage, and
//! aliasing before dispatch (see its `resources.rs::forward_aliases` /
//! `backward_aliases`, mirrored below), and those checks already run through
//! the shared, backend-neutral `hephaestus_core::plan_convolution_*`
//! functions this module calls. Once a plan validates, dispatch delegates
//! straight to leto-ops' `convolution_forward_into` /
//! `convolution_backward_accumulate` / `convolution_transposed_forward_into`
//! / `convolution_transposed_backward_accumulate`, which perform the real
//! arithmetic and are the only code paths that touch a destination buffer.

use eunomia::Pod;
use hephaestus_core::{
    ConvolutionBackwardOperands, ConvolutionForwardOperands, ConvolutionOps, Result,
    plan_convolution_backward, plan_convolution_forward, plan_transposed_convolution_backward,
    plan_transposed_convolution_forward,
};
use leto::{ArrayView, ArrayViewMut, ConvolutionParameters, TransposedConvolutionParameters};
use leto_ops::{Scalar, TransposedConvolutionGradients};

use crate::operands::with_operand_reads;
use crate::{HostBuffer, HostDevice, map_leto_error};

/// Regular and transposed convolution for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostConvolutionOps;

/// Prepared host regular-convolution forward resources.
pub struct HostConvolutionForward<'a, T, const R: usize, const S: usize> {
    operands: ConvolutionForwardOperands<'a, HostBuffer<T>, R>,
    parameters: ConvolutionParameters<S>,
}

/// Prepared host regular-convolution backward resources.
pub struct HostConvolutionBackward<'a, T, const R: usize, const S: usize> {
    operands: ConvolutionBackwardOperands<'a, HostBuffer<T>, R>,
    parameters: ConvolutionParameters<S>,
}

/// Prepared host transposed-convolution forward resources.
pub struct HostConvolutionTransposedForward<'a, T, const R: usize, const S: usize> {
    operands: ConvolutionForwardOperands<'a, HostBuffer<T>, R>,
    parameters: TransposedConvolutionParameters<S>,
}

/// Prepared host transposed-convolution backward resources.
pub struct HostConvolutionTransposedBackward<'a, T, const R: usize, const S: usize> {
    operands: ConvolutionBackwardOperands<'a, HostBuffer<T>, R>,
    parameters: TransposedConvolutionParameters<S>,
}

impl<T> ConvolutionOps<HostDevice, T> for HostConvolutionOps
where
    T: Pod + Scalar,
{
    type PreparedForward<'a, const R: usize, const S: usize>
        = HostConvolutionForward<'a, T, R, S>
    where
        HostDevice: 'a,
        T: 'a;
    type PreparedBackward<'a, const R: usize, const S: usize>
        = HostConvolutionBackward<'a, T, R, S>
    where
        HostDevice: 'a,
        T: 'a;
    type PreparedTransposedForward<'a, const R: usize, const S: usize>
        = HostConvolutionTransposedForward<'a, T, R, S>
    where
        HostDevice: 'a,
        T: 'a;
    type PreparedTransposedBackward<'a, const R: usize, const S: usize>
        = HostConvolutionTransposedBackward<'a, T, R, S>
    where
        HostDevice: 'a,
        T: 'a;

    fn prepare_convolution_forward<'a, const R: usize, const S: usize>(
        &self,
        _device: &'a HostDevice,
        operands: ConvolutionForwardOperands<'a, HostBuffer<T>, R>,
        parameters: ConvolutionParameters<S>,
    ) -> Result<Self::PreparedForward<'a, R, S>> {
        plan_convolution_forward::<T, _, R, S>(&operands, parameters, forward_aliases(&operands))?;
        Ok(HostConvolutionForward {
            operands,
            parameters,
        })
    }

    fn dispatch_convolution_forward<const R: usize, const S: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedForward<'_, R, S>,
    ) -> Result<()> {
        dispatch_forward(&prepared.operands, |input, weight, bias, output| {
            leto_ops::convolution_forward_into(input, weight, bias, prepared.parameters, output)
        })
    }

    fn prepare_convolution_backward<'a, const R: usize, const S: usize>(
        &self,
        _device: &'a HostDevice,
        operands: ConvolutionBackwardOperands<'a, HostBuffer<T>, R>,
        parameters: ConvolutionParameters<S>,
    ) -> Result<Self::PreparedBackward<'a, R, S>> {
        plan_convolution_backward::<T, _, R, S>(
            &operands,
            parameters,
            backward_aliases(&operands),
        )?;
        Ok(HostConvolutionBackward {
            operands,
            parameters,
        })
    }

    fn dispatch_convolution_backward<const R: usize, const S: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedBackward<'_, R, S>,
    ) -> Result<()> {
        dispatch_backward(
            &prepared.operands,
            |input, weight, grad_output, grad_input, grad_weight, grad_bias| {
                leto_ops::convolution_backward_accumulate(
                    input,
                    weight,
                    grad_output,
                    prepared.parameters,
                    grad_input,
                    grad_weight,
                    grad_bias,
                )
            },
        )
    }

    fn prepare_convolution_transposed_forward<'a, const R: usize, const S: usize>(
        &self,
        _device: &'a HostDevice,
        operands: ConvolutionForwardOperands<'a, HostBuffer<T>, R>,
        parameters: TransposedConvolutionParameters<S>,
    ) -> Result<Self::PreparedTransposedForward<'a, R, S>> {
        plan_transposed_convolution_forward::<T, _, R, S>(
            &operands,
            parameters,
            forward_aliases(&operands),
        )?;
        Ok(HostConvolutionTransposedForward {
            operands,
            parameters,
        })
    }

    fn dispatch_convolution_transposed_forward<const R: usize, const S: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedTransposedForward<'_, R, S>,
    ) -> Result<()> {
        dispatch_forward(&prepared.operands, |input, weight, bias, output| {
            leto_ops::convolution_transposed_forward_into(
                input,
                weight,
                bias,
                prepared.parameters,
                output,
            )
        })
    }

    fn prepare_convolution_transposed_backward<'a, const R: usize, const S: usize>(
        &self,
        _device: &'a HostDevice,
        operands: ConvolutionBackwardOperands<'a, HostBuffer<T>, R>,
        parameters: TransposedConvolutionParameters<S>,
    ) -> Result<Self::PreparedTransposedBackward<'a, R, S>> {
        plan_transposed_convolution_backward::<T, _, R, S>(
            &operands,
            parameters,
            backward_aliases(&operands),
        )?;
        Ok(HostConvolutionTransposedBackward {
            operands,
            parameters,
        })
    }

    fn dispatch_convolution_transposed_backward<const R: usize, const S: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedTransposedBackward<'_, R, S>,
    ) -> Result<()> {
        dispatch_backward(
            &prepared.operands,
            |input, weight, grad_output, grad_input, grad_weight, grad_bias| {
                leto_ops::convolution_transposed_backward_accumulate(
                    input,
                    weight,
                    grad_output,
                    prepared.parameters,
                    TransposedConvolutionGradients::new(grad_input, grad_weight, grad_bias),
                )
            },
        )
    }
}

/// Read `input`/`weight`/optional `bias`, deduplicating guards for aliased
/// operands, then run `compute` against a freshly locked `output` and map
/// its error. Shared by the regular and transposed forward dispatch paths,
/// which differ only in which leto-ops kernel `compute` calls.
fn dispatch_forward<T, const R: usize>(
    operands: &ConvolutionForwardOperands<'_, HostBuffer<T>, R>,
    compute: impl FnOnce(
        &ArrayView<'_, T, R>,
        &ArrayView<'_, T, R>,
        Option<&ArrayView<'_, T, 1>>,
        &mut ArrayViewMut<'_, T, R>,
    ) -> leto::Result<()>,
) -> Result<()>
where
    T: Pod,
{
    let mut reads: Vec<&HostBuffer<T>> = vec![operands.input.buffer, operands.weight.buffer];
    if let Some(bias) = operands.bias {
        reads.push(bias.buffer);
    }
    with_operand_reads(&reads, |slices| -> Result<()> {
        let input_view =
            ArrayView::try_new(*operands.input.layout, slices[0]).map_err(map_leto_error)?;
        let weight_view =
            ArrayView::try_new(*operands.weight.layout, slices[1]).map_err(map_leto_error)?;
        let bias_view = operands
            .bias
            .map(|bias| ArrayView::try_new(*bias.layout, slices[2]).map_err(map_leto_error))
            .transpose()?;
        let mut output_cells = operands.output.buffer.write();
        let mut output_view = ArrayViewMut::try_new(*operands.output.layout, &mut output_cells)
            .map_err(map_leto_error)?;
        compute(
            &input_view,
            &weight_view,
            bias_view.as_ref(),
            &mut output_view,
        )
        .map_err(map_leto_error)
    })
}

/// Read `input`/`weight`/`grad_output`, deduplicating guards for aliased
/// operands, open one write guard per requested gradient target, then run
/// `compute` and map its error. Shared by the regular and transposed
/// backward dispatch paths, which differ only in which leto-ops kernel
/// `compute` calls and how it bundles the gradient targets.
fn dispatch_backward<T, const R: usize>(
    operands: &ConvolutionBackwardOperands<'_, HostBuffer<T>, R>,
    compute: impl for<'v> FnOnce(
        &ArrayView<'_, T, R>,
        &ArrayView<'_, T, R>,
        &ArrayView<'_, T, R>,
        Option<&mut ArrayViewMut<'v, T, R>>,
        Option<&mut ArrayViewMut<'v, T, R>>,
        Option<&mut ArrayViewMut<'v, T, 1>>,
    ) -> leto::Result<()>,
) -> Result<()>
where
    T: Pod,
{
    let reads: Vec<&HostBuffer<T>> = vec![
        operands.input.buffer,
        operands.weight.buffer,
        operands.grad_output.buffer,
    ];
    with_operand_reads(&reads, |slices| -> Result<()> {
        let input_view =
            ArrayView::try_new(*operands.input.layout, slices[0]).map_err(map_leto_error)?;
        let weight_view =
            ArrayView::try_new(*operands.weight.layout, slices[1]).map_err(map_leto_error)?;
        let grad_output_view =
            ArrayView::try_new(*operands.grad_output.layout, slices[2]).map_err(map_leto_error)?;

        let mut input_grad_cells = operands.gradients.input.map(|target| target.buffer.write());
        let mut weight_grad_cells = operands
            .gradients
            .weight
            .map(|target| target.buffer.write());
        let mut bias_grad_cells = operands.gradients.bias.map(|target| target.buffer.write());

        let mut input_grad_view = match (&mut input_grad_cells, operands.gradients.input) {
            (Some(cells), Some(target)) => {
                Some(ArrayViewMut::try_new(*target.layout, cells).map_err(map_leto_error)?)
            }
            _ => None,
        };
        let mut weight_grad_view = match (&mut weight_grad_cells, operands.gradients.weight) {
            (Some(cells), Some(target)) => {
                Some(ArrayViewMut::try_new(*target.layout, cells).map_err(map_leto_error)?)
            }
            _ => None,
        };
        let mut bias_grad_view = match (&mut bias_grad_cells, operands.gradients.bias) {
            (Some(cells), Some(target)) => {
                Some(ArrayViewMut::try_new(*target.layout, cells).map_err(map_leto_error)?)
            }
            _ => None,
        };

        compute(
            &input_view,
            &weight_view,
            &grad_output_view,
            input_grad_view.as_mut(),
            weight_grad_view.as_mut(),
            bias_grad_view.as_mut(),
        )
        .map_err(map_leto_error)
    })
}

/// Illegal-aliasing predicate mirroring `hephaestus-wgpu`'s
/// `application/convolution/resources.rs::forward_aliases`.
fn forward_aliases<T, const R: usize>(
    operands: &ConvolutionForwardOperands<'_, HostBuffer<T>, R>,
) -> bool {
    operands.output.buffer.aliases(operands.input.buffer)
        || operands.output.buffer.aliases(operands.weight.buffer)
        || operands
            .bias
            .is_some_and(|bias| operands.output.buffer.aliases(bias.buffer))
}

/// Illegal-aliasing predicate mirroring `hephaestus-wgpu`'s
/// `application/convolution/resources.rs::backward_aliases`.
fn backward_aliases<T, const R: usize>(
    operands: &ConvolutionBackwardOperands<'_, HostBuffer<T>, R>,
) -> bool {
    let reads = [
        operands.input.buffer,
        operands.weight.buffer,
        operands.grad_output.buffer,
    ];
    let input = operands.gradients.input.map(|view| view.buffer);
    let weight = operands.gradients.weight.map(|view| view.buffer);
    let bias = operands.gradients.bias.map(|view| view.buffer);
    let target_reads_alias = [input, weight, bias]
        .into_iter()
        .flatten()
        .any(|target| reads.into_iter().any(|read| target.aliases(read)));
    let targets_alias = input.is_some_and(|input| {
        weight.is_some_and(|weight| input.aliases(weight))
            || bias.is_some_and(|bias| input.aliases(bias))
    }) || weight
        .is_some_and(|weight| bias.is_some_and(|bias| weight.aliases(bias)));
    target_reads_alias || targets_alias
}
