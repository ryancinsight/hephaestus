//! Leto-backed host implementation of the unfold/fold seam.

use bytemuck::Pod;
use hephaestus_core::{
    SlidingWindowFoldOperands, SlidingWindowOps, SlidingWindowUnfoldOperands,
    plan_sliding_window_fold, plan_sliding_window_unfold,
};
use leto::{ArrayView, ArrayViewMut, WindowParameters};
use leto_ops::Scalar;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Prepared host unfold resources.
#[derive(Clone, Copy)]
pub struct HostSlidingWindowUnfold<'a, T, const R: usize, const S: usize> {
    operands: SlidingWindowUnfoldOperands<'a, HostBuffer<T>, R>,
    parameters: WindowParameters<S>,
}

/// Prepared host fold resources.
#[derive(Clone, Copy)]
pub struct HostSlidingWindowFold<'a, T, const R: usize, const S: usize> {
    operands: SlidingWindowFoldOperands<'a, HostBuffer<T>, R>,
    output_spatial_shape: [usize; S],
    parameters: WindowParameters<S>,
}

/// Leto-backed host unfold/fold operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostSlidingWindowOps;

impl<T> SlidingWindowOps<HostDevice, T> for HostSlidingWindowOps
where
    T: Pod + Scalar,
{
    type PreparedUnfold<'a, const R: usize, const S: usize>
        = HostSlidingWindowUnfold<'a, T, R, S>
    where
        HostDevice: 'a,
        T: 'a;
    type PreparedFold<'a, const R: usize, const S: usize>
        = HostSlidingWindowFold<'a, T, R, S>
    where
        HostDevice: 'a,
        T: 'a;

    fn prepare_unfold<'a, const R: usize, const S: usize>(
        &self,
        _device: &'a HostDevice,
        operands: SlidingWindowUnfoldOperands<'a, HostBuffer<T>, R>,
        parameters: WindowParameters<S>,
    ) -> hephaestus_core::Result<Self::PreparedUnfold<'a, R, S>> {
        let illegal_aliasing = operands.input.buffer.aliases(operands.output.buffer);
        plan_sliding_window_unfold::<T, _, R, S>(&operands, parameters, illegal_aliasing)?;
        Ok(HostSlidingWindowUnfold {
            operands,
            parameters,
        })
    }

    fn dispatch_unfold<const R: usize, const S: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedUnfold<'_, R, S>,
    ) -> hephaestus_core::Result<()> {
        let input = prepared.operands.input;
        let output = prepared.operands.output;
        let input_cells = input.buffer.read();
        let mut output_cells = output.buffer.write();
        let input_view = ArrayView::new(*input.layout, &input_cells);
        let mut output_view = ArrayViewMut::new(*output.layout, &mut output_cells);
        leto_ops::unfold_into(&input_view, prepared.parameters, &mut output_view)
            .map_err(map_leto_error)
    }

    fn prepare_fold<'a, const R: usize, const S: usize>(
        &self,
        _device: &'a HostDevice,
        operands: SlidingWindowFoldOperands<'a, HostBuffer<T>, R>,
        output_spatial_shape: [usize; S],
        parameters: WindowParameters<S>,
    ) -> hephaestus_core::Result<Self::PreparedFold<'a, R, S>> {
        let illegal_aliasing = operands.input.buffer.aliases(operands.output.buffer);
        plan_sliding_window_fold::<T, _, R, S>(
            &operands,
            output_spatial_shape,
            parameters,
            illegal_aliasing,
        )?;
        Ok(HostSlidingWindowFold {
            operands,
            output_spatial_shape,
            parameters,
        })
    }

    fn dispatch_fold<const R: usize, const S: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedFold<'_, R, S>,
    ) -> hephaestus_core::Result<()> {
        let input = prepared.operands.input;
        let output = prepared.operands.output;
        let input_cells = input.buffer.read();
        let mut output_cells = output.buffer.write();
        let input_view = ArrayView::new(*input.layout, &input_cells);
        let mut output_view = ArrayViewMut::new(*output.layout, &mut output_cells);
        leto_ops::fold_into(
            &input_view,
            prepared.output_spatial_shape,
            prepared.parameters,
            &mut output_view,
        )
        .map_err(map_leto_error)
    }
}

#[cfg(test)]
mod tests {
    use super::HostSlidingWindowOps;
    use hephaestus_core::{
        ComputeDevice, SlidingWindowFoldOperands, SlidingWindowOps, SlidingWindowUnfoldOperands,
        StridedView,
    };
    use leto::{Layout, WindowParameters};

    #[test]
    fn host_unfold_and_fold_match_the_adjoint_reference() {
        let device = super::HostDevice::new();
        let input = device
            .upload(&[1_i32, 2, 3, 4])
            .expect("input upload succeeds");
        let unfolded = device
            .alloc_zeroed::<i32>(6)
            .expect("unfold allocation succeeds");
        let folded = device
            .alloc_zeroed::<i32>(4)
            .expect("fold allocation succeeds");
        let input_layout = Layout::c_contiguous([1, 1, 4]).expect("input layout");
        let unfolded_layout = Layout::c_contiguous([1, 2, 3]).expect("unfold layout");
        let folded_layout = Layout::c_contiguous([1, 1, 4]).expect("fold layout");
        let parameters =
            WindowParameters::new([2], [1], [0], [1]).expect("valid sliding-window parameters");

        HostSlidingWindowOps
            .unfold_into(
                &device,
                SlidingWindowUnfoldOperands {
                    input: StridedView::new(&input, &input_layout),
                    output: StridedView::new(&unfolded, &unfolded_layout),
                },
                parameters,
            )
            .expect("host unfold succeeds");
        HostSlidingWindowOps
            .fold_into(
                &device,
                SlidingWindowFoldOperands {
                    input: StridedView::new(&unfolded, &unfolded_layout),
                    output: StridedView::new(&folded, &folded_layout),
                },
                [4],
                parameters,
            )
            .expect("host fold succeeds");

        let mut unfolded_actual = [0_i32; 6];
        let mut folded_actual = [0_i32; 4];
        device
            .download(&unfolded, &mut unfolded_actual)
            .expect("unfold download succeeds");
        device
            .download(&folded, &mut folded_actual)
            .expect("fold download succeeds");
        assert_eq!(unfolded_actual, [1, 2, 3, 2, 3, 4]);
        assert_eq!(folded_actual, [1, 4, 6, 4]);
    }
}
