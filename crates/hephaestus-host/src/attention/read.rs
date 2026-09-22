//! Element access and status folding shared by the two preflight passes.

use core::num::NonZeroUsize;
use hephaestus_core::{AttentionPlan, AttentionSemanticStatus};

use leto::ArrayView;

/// Map the `u32::MAX` no-failure sentinel used by the status-reduction
/// helpers below to [`AttentionSemanticStatus::Valid`]'s protocol code,
/// mirroring `hephaestus-wgpu`'s preflight status buffers, which reset to
/// `u32::MAX` before dispatch and are read back the same way.
pub(super) fn finalize_status(status: u32) -> u32 {
    if status == u32::MAX {
        AttentionSemanticStatus::Valid.code()
    } else {
        status
    }
}

pub(super) fn all_finite<const N: usize>(view: &ArrayView<'_, f32, N>) -> bool {
    view.iter().all(|value| value.is_finite())
}

pub(super) fn read3(view: &ArrayView<'_, f32, 3>, batch: usize, row: usize, col: usize) -> f32 {
    *view
        .get([batch, row, col])
        .expect("invariant: plan-validated attention index is in bounds")
}

pub(super) fn read2(view: &ArrayView<'_, f32, 2>, row: usize, col: usize) -> f32 {
    *view
        .get([row, col])
        .expect("invariant: plan-validated attention mask index is in bounds")
}

/// Whether `(batch, query_index, key_index)` survives causal and keep-mask
/// gating, mirroring `hephaestus-wgpu`'s `shader/preflight.rs::kept`.
pub(super) fn is_kept(
    grouped: Option<(ArrayView<'_, f32, 2>, NonZeroUsize)>,
    causal: bool,
    batch: usize,
    query_index: usize,
    key_index: usize,
) -> bool {
    if causal && key_index > query_index {
        return false;
    }
    match grouped {
        None => true,
        Some((view, heads_per_batch)) => {
            let mask_batch = batch / heads_per_batch.get();
            read2(&view, mask_batch, key_index) != 0.0
        }
    }
}

/// Score and dot-product accumulation, checked for finiteness at every step
/// exactly as `hephaestus-wgpu`'s `attention_score` does.
pub(super) fn attention_score(
    query: &ArrayView<'_, f32, 3>,
    key: &ArrayView<'_, f32, 3>,
    batch: usize,
    query_index: usize,
    key_index: usize,
    key_feature: usize,
    scale: f32,
) -> (f32, bool) {
    let mut dot = 0.0_f32;
    let mut nonfinite = false;
    for feature in 0..key_feature {
        dot += read3(query, batch, query_index, feature) * read3(key, batch, key_index, feature);
        nonfinite |= !dot.is_finite();
    }
    let scaled = dot * scale;
    nonfinite |= !scaled.is_finite();
    (scaled, nonfinite)
}

pub(super) fn score_index(
    plan: AttentionPlan,
    batch: usize,
    query_index: usize,
    key_index: usize,
) -> usize {
    (batch * plan.query_sequence + query_index) * plan.key_sequence + key_index
}
