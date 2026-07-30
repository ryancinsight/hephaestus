use hephaestus_core::{
    ConvolutionBackwardOperands, ConvolutionForwardOperands, ConvolutionGradientViews, StridedView,
};
use hephaestus_wgpu::WgpuBuffer;

use crate::MetalBuffer;

pub(super) fn forward<'a, const R: usize>(
    operands: ConvolutionForwardOperands<'a, MetalBuffer<f32>, R>,
) -> ConvolutionForwardOperands<'a, WgpuBuffer<f32>, R> {
    ConvolutionForwardOperands {
        input: view(operands.input),
        weight: view(operands.weight),
        bias: operands.bias.map(view),
        output: view(operands.output),
    }
}

pub(super) fn backward<'a, const R: usize>(
    operands: ConvolutionBackwardOperands<'a, MetalBuffer<f32>, R>,
) -> ConvolutionBackwardOperands<'a, WgpuBuffer<f32>, R> {
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

fn view<const R: usize>(
    view: StridedView<'_, MetalBuffer<f32>, R>,
) -> StridedView<'_, WgpuBuffer<f32>, R> {
    StridedView::new(view.buffer.wgpu_buffer(), view.layout)
}
