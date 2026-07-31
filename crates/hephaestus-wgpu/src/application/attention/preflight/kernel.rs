use core::any::TypeId;

use hephaestus_core::{AttentionSemanticStatus, Result};

use super::super::metadata::AttentionMeta;
use super::super::prepared::PreparedAttentionKernel;
use super::super::resources::binding;
use super::super::seam::{WORKGROUP_WIDTH, prepare};
use super::super::shader::finite_preflight_shader;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

#[expect(
    clippy::too_many_arguments,
    reason = "finite preflight preparation enumerates the device ABI and semantic status"
)]
pub(super) fn prepare_finite<K: 'static>(
    device: &WgpuDevice,
    metadata: &AttentionMeta,
    elements: usize,
    layout: &'static str,
    rank: u32,
    failure: AttentionSemanticStatus,
    source: &WgpuBuffer<f32>,
    status: &WgpuBuffer<u32>,
) -> Result<PreparedAttentionKernel> {
    prepare_status::<K>(
        device,
        metadata,
        elements,
        "hephaestus-attention-finite-preflight",
        &[binding(0, source), binding(1, status)],
        || finite_preflight_shader(layout, rank, failure, WORKGROUP_WIDTH.get()),
    )
}

pub(super) fn prepare_status<K: 'static>(
    device: &WgpuDevice,
    metadata: &AttentionMeta,
    elements: usize,
    label: &'static str,
    storage_entries: &[wgpu::BindGroupEntry<'_>],
    shader: impl FnOnce() -> String,
) -> Result<PreparedAttentionKernel> {
    prepare(
        device,
        metadata,
        elements,
        label,
        storage_entries,
        shader,
        TypeId::of::<K>(),
    )
}
