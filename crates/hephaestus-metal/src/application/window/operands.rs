use hephaestus_core::{
    PoolingBackwardOperands, PoolingForwardOperands, SlidingWindowFoldOperands,
    SlidingWindowUnfoldOperands, StridedView,
};
use hephaestus_wgpu::WgpuBuffer;

use crate::MetalBuffer;

pub(super) fn pooling_forward<'a, T, const R: usize>(
    operands: PoolingForwardOperands<'a, MetalBuffer<T>, R>,
) -> PoolingForwardOperands<'a, WgpuBuffer<T>, R> {
    PoolingForwardOperands {
        input: view(operands.input),
        output: view(operands.output),
    }
}

pub(super) fn pooling_backward<'a, T, const R: usize>(
    operands: PoolingBackwardOperands<'a, MetalBuffer<T>, R>,
) -> PoolingBackwardOperands<'a, WgpuBuffer<T>, R> {
    PoolingBackwardOperands {
        input: operands.input.map(view),
        grad_output: view(operands.grad_output),
        grad_input: view(operands.grad_input),
    }
}

pub(super) fn unfold<'a, T, const R: usize>(
    operands: SlidingWindowUnfoldOperands<'a, MetalBuffer<T>, R>,
) -> SlidingWindowUnfoldOperands<'a, WgpuBuffer<T>, R> {
    SlidingWindowUnfoldOperands {
        input: view(operands.input),
        output: view(operands.output),
    }
}

pub(super) fn fold<'a, T, const R: usize>(
    operands: SlidingWindowFoldOperands<'a, MetalBuffer<T>, R>,
) -> SlidingWindowFoldOperands<'a, WgpuBuffer<T>, R> {
    SlidingWindowFoldOperands {
        input: view(operands.input),
        output: view(operands.output),
    }
}

fn view<'a, T, const R: usize>(
    view: StridedView<'a, MetalBuffer<T>, R>,
) -> StridedView<'a, WgpuBuffer<T>, R> {
    StridedView::new(view.buffer.wgpu_buffer(), view.layout)
}
