//! Leto-backed host implementation of the pooling seam.

use bytemuck::Pod;
use hephaestus_core::{
    PoolingBackwardOperands, PoolingForwardOperands, PoolingMode, PoolingOps,
    plan_pooling_backward, plan_pooling_forward,
};
use leto::{ArrayView, ArrayViewMut, WindowParameters};
use leto_ops::Scalar;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Prepared host pooling forward resources.
#[derive(Clone, Copy)]
pub struct HostPoolingForward<'a, T, const R: usize, const S: usize> {
    operands: PoolingForwardOperands<'a, HostBuffer<T>, R>,
    parameters: WindowParameters<S>,
    mode: PoolingMode,
}

/// Prepared host pooling backward resources.
#[derive(Clone, Copy)]
pub struct HostPoolingBackward<'a, T, const R: usize, const S: usize> {
    operands: PoolingBackwardOperands<'a, HostBuffer<T>, R>,
    parameters: WindowParameters<S>,
    mode: PoolingMode,
}

/// Leto-backed host pooling operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostPoolingOps;

impl<T> PoolingOps<HostDevice, T> for HostPoolingOps
where
    T: Pod + Scalar,
{
    type PreparedForward<'a, const R: usize, const S: usize>
        = HostPoolingForward<'a, T, R, S>
    where
        HostDevice: 'a,
        T: 'a;
    type PreparedBackward<'a, const R: usize, const S: usize>
        = HostPoolingBackward<'a, T, R, S>
    where
        HostDevice: 'a,
        T: 'a;

    fn prepare_pooling_forward<'a, const R: usize, const S: usize>(
        &self,
        _device: &'a HostDevice,
        operands: PoolingForwardOperands<'a, HostBuffer<T>, R>,
        parameters: WindowParameters<S>,
        mode: PoolingMode,
    ) -> hephaestus_core::Result<Self::PreparedForward<'a, R, S>> {
        let illegal_aliasing = operands.input.buffer.aliases(operands.output.buffer);
        plan_pooling_forward::<T, _, R, S>(&operands, parameters, illegal_aliasing)?;
        Ok(HostPoolingForward {
            operands,
            parameters,
            mode,
        })
    }

    fn dispatch_pooling_forward<const R: usize, const S: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedForward<'_, R, S>,
    ) -> hephaestus_core::Result<()> {
        let input = prepared.operands.input;
        let output = prepared.operands.output;
        let input_cells = input.buffer.read();
        let mut output_cells = output.buffer.write();
        let input_view = ArrayView::new(*input.layout, &input_cells);
        let mut output_view = ArrayViewMut::new(*output.layout, &mut output_cells);
        leto_ops::pooling_forward_into(
            &input_view,
            prepared.parameters,
            leto_mode(prepared.mode),
            &mut output_view,
        )
        .map_err(map_leto_error)
    }

    fn prepare_pooling_backward<'a, const R: usize, const S: usize>(
        &self,
        _device: &'a HostDevice,
        operands: PoolingBackwardOperands<'a, HostBuffer<T>, R>,
        parameters: WindowParameters<S>,
        mode: PoolingMode,
    ) -> hephaestus_core::Result<Self::PreparedBackward<'a, R, S>> {
        let illegal_aliasing = operands.grad_input.buffer.aliases(operands.input.buffer)
            || operands
                .grad_input
                .buffer
                .aliases(operands.grad_output.buffer);
        plan_pooling_backward::<T, _, R, S>(&operands, parameters, illegal_aliasing)?;
        Ok(HostPoolingBackward {
            operands,
            parameters,
            mode,
        })
    }

    fn dispatch_pooling_backward<const R: usize, const S: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedBackward<'_, R, S>,
    ) -> hephaestus_core::Result<()> {
        let input = prepared.operands.input;
        let grad_output = prepared.operands.grad_output;
        let grad_input = prepared.operands.grad_input;
        let input_cells = input.buffer.read();
        let grad_output_cells = grad_output.buffer.read();
        let mut grad_input_cells = grad_input.buffer.write();
        let input_view = ArrayView::new(*input.layout, &input_cells);
        let grad_output_view = ArrayView::new(*grad_output.layout, &grad_output_cells);
        let mut grad_input_view = ArrayViewMut::new(*grad_input.layout, &mut grad_input_cells);
        leto_ops::pooling_backward_accumulate(
            &grad_output_view,
            &input_view,
            prepared.parameters,
            leto_mode(prepared.mode),
            &mut grad_input_view,
        )
        .map_err(map_leto_error)
    }
}

fn leto_mode(mode: PoolingMode) -> leto_ops::PoolingMode {
    match mode {
        PoolingMode::Maximum => leto_ops::PoolingMode::Maximum,
        PoolingMode::Average => leto_ops::PoolingMode::Average,
    }
}

#[cfg(test)]
mod tests {
    use super::HostPoolingOps;
    use hephaestus_core::{
        ComputeDevice, PoolingBackwardOperands, PoolingForwardOperands, PoolingMode, PoolingOps,
        StridedView,
    };
    use leto::{Layout, WindowParameters};

    fn parameters() -> WindowParameters<2> {
        WindowParameters::new([2, 2], [1, 1], [0, 0], [1, 1]).expect("valid pooling parameters")
    }

    #[test]
    fn host_forward_matches_pooling_reference() {
        let device = super::HostDevice::new();
        let input = device
            .upload(&[1_i32, 2, 3, 4, 5, 6, 7, 8, 9])
            .expect("input upload succeeds");
        let output = device
            .alloc_zeroed::<i32>(4)
            .expect("output allocation succeeds");
        let input_layout = Layout::c_contiguous([1, 1, 3, 3]).expect("input layout");
        let output_layout = Layout::c_contiguous([1, 1, 2, 2]).expect("output layout");

        HostPoolingOps
            .pooling_forward_into(
                &device,
                PoolingForwardOperands {
                    input: StridedView::new(&input, &input_layout),
                    output: StridedView::new(&output, &output_layout),
                },
                parameters(),
                PoolingMode::Maximum,
            )
            .expect("host pooling forward succeeds");

        let mut actual = [0_i32; 4];
        device
            .download(&output, &mut actual)
            .expect("output download succeeds");
        assert_eq!(actual, [5, 6, 8, 9]);
    }

    #[test]
    fn host_backward_accumulates_average_gradient() {
        let device = super::HostDevice::new();
        let input = device
            .upload(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .expect("input upload succeeds");
        let grad_output = device
            .upload(&[1.0_f64; 4])
            .expect("gradient upload succeeds");
        let grad_input = device
            .upload(&[0.0_f64; 9])
            .expect("gradient allocation succeeds");
        let input_layout = Layout::c_contiguous([1, 1, 3, 3]).expect("input layout");
        let output_layout = Layout::c_contiguous([1, 1, 2, 2]).expect("output layout");

        HostPoolingOps
            .pooling_backward_accumulate(
                &device,
                PoolingBackwardOperands {
                    input: StridedView::new(&input, &input_layout),
                    grad_output: StridedView::new(&grad_output, &output_layout),
                    grad_input: StridedView::new(&grad_input, &input_layout),
                },
                parameters(),
                PoolingMode::Average,
            )
            .expect("host pooling backward succeeds");

        let mut actual = [0.0_f64; 9];
        device
            .download(&grad_input, &mut actual)
            .expect("gradient download succeeds");
        assert_eq!(actual, [0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25]);
    }
}
