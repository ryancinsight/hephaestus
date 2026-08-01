use hephaestus_core::{
    ConvolutionBackwardOperands, ConvolutionForwardOperands, ConvolutionGradientViews, StridedView,
};
use hephaestus_wgpu::WgpuBuffer;

use crate::MetalBuffer;

pub(super) fn forward<'a, T, const R: usize>(
    operands: ConvolutionForwardOperands<'a, MetalBuffer<T>, R>,
) -> ConvolutionForwardOperands<'a, WgpuBuffer<T>, R> {
    ConvolutionForwardOperands {
        input: view(operands.input),
        weight: view(operands.weight),
        bias: operands.bias.map(view),
        output: view(operands.output),
    }
}

pub(super) fn backward<'a, T, const R: usize>(
    operands: ConvolutionBackwardOperands<'a, MetalBuffer<T>, R>,
) -> ConvolutionBackwardOperands<'a, WgpuBuffer<T>, R> {
    ConvolutionBackwardOperands {
        input: view(operands.input),
        weight: view(operands.weight),
        grad_output: view(operands.grad_output),
        gradients: ConvolutionGradientViews {
            input: operands.gradients.input.map(view),
            weight: operands.gradients.weight.map(view),
            bias: operands.gradients.bias.map(view),
        },
    }
}

fn view<T, const R: usize>(
    view: StridedView<'_, MetalBuffer<T>, R>,
) -> StridedView<'_, WgpuBuffer<T>, R> {
    StridedView::new(view.buffer.wgpu_buffer(), view.layout)
}
