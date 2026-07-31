use hephaestus_core::{
    AttentionBackwardOperands, AttentionCausality, AttentionForwardOperands,
    AttentionGradientViews, AttentionMask, GroupedKeepMask, StridedView,
};
use hephaestus_wgpu::WgpuBuffer;

use crate::MetalBuffer;

pub(super) fn forward(
    operands: AttentionForwardOperands<'_, MetalBuffer<f32>, f32>,
) -> AttentionForwardOperands<'_, WgpuBuffer<f32>, f32> {
    AttentionForwardOperands {
        query: view(operands.query),
        key: view(operands.key),
        value: view(operands.value),
        mask: mask(operands.mask),
        scale: operands.scale,
        output: view(operands.output),
        weights: view(operands.weights),
    }
}

pub(super) fn backward(
    operands: AttentionBackwardOperands<'_, MetalBuffer<f32>, f32>,
) -> AttentionBackwardOperands<'_, WgpuBuffer<f32>, f32> {
    AttentionBackwardOperands {
        grad_output: view(operands.grad_output),
        query: view(operands.query),
        key: view(operands.key),
        value: view(operands.value),
        weights: view(operands.weights),
        scale: operands.scale,
        gradients: AttentionGradientViews {
            query: operands.gradients.query.map(view),
            key: operands.gradients.key.map(view),
            value: operands.gradients.value.map(view),
        },
    }
}

fn mask(source: AttentionMask<'_, MetalBuffer<f32>>) -> AttentionMask<'_, WgpuBuffer<f32>> {
    let keep = source
        .grouped_keep()
        .map(|keep| GroupedKeepMask::new(view(keep.view()), keep.heads_per_batch()));
    match (source.causality(), keep) {
        (AttentionCausality::Unrestricted, None) => AttentionMask::unrestricted(),
        (AttentionCausality::Causal, None) => AttentionMask::causal(),
        (AttentionCausality::Unrestricted, Some(keep)) => AttentionMask::keep(keep),
        (AttentionCausality::Causal, Some(keep)) => AttentionMask::causal_keep(keep),
    }
}

fn view<T, const R: usize>(
    source: StridedView<'_, MetalBuffer<T>, R>,
) -> StridedView<'_, WgpuBuffer<T>, R> {
    StridedView::new(source.buffer.wgpu_buffer(), source.layout)
}
