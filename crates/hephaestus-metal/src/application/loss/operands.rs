use hephaestus_core::{CrossEntropyBackwardOperands, CrossEntropyForwardOperands, StridedView};
use hephaestus_wgpu::WgpuBuffer;

use crate::MetalBuffer;

pub(super) fn forward(
    operands: CrossEntropyForwardOperands<'_, MetalBuffer<f32>, MetalBuffer<u32>>,
) -> CrossEntropyForwardOperands<'_, WgpuBuffer<f32>, WgpuBuffer<u32>> {
    CrossEntropyForwardOperands {
        logits: view(operands.logits),
        targets: view(operands.targets),
        loss: view(operands.loss),
        probabilities: view(operands.probabilities),
    }
}

pub(super) fn backward(
    operands: CrossEntropyBackwardOperands<'_, MetalBuffer<f32>, MetalBuffer<u32>>,
) -> CrossEntropyBackwardOperands<'_, WgpuBuffer<f32>, WgpuBuffer<u32>> {
    CrossEntropyBackwardOperands {
        output_gradient: view(operands.output_gradient),
        probabilities: view(operands.probabilities),
        targets: view(operands.targets),
        logit_gradient: view(operands.logit_gradient),
    }
}

fn view<T, const R: usize>(
    source: StridedView<'_, MetalBuffer<T>, R>,
) -> StridedView<'_, WgpuBuffer<T>, R> {
    StridedView::new(source.buffer.wgpu_buffer(), source.layout)
}
