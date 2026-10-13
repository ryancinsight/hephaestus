use hephaestus_core::Result;

use crate::application::prepared::{checked_submit, validate_device_owner};
use crate::infrastructure::device::WgpuDevice;

use super::types::PreparedElementwise;

pub(crate) fn dispatch_prepared<const N: usize>(
    device: &WgpuDevice,
    prepared: &PreparedElementwise,
    label: &'static str,
) -> Result<()> {
    validate_device_owner(&prepared.owner, device, "elementwise operation")?;
    let Some(pipeline) = &prepared.pipeline else {
        return Ok(());
    };
    let Some(bind_group) = &prepared.bind_group else {
        return Ok(());
    };

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    crate::application::pipeline::encode_compute_pass(
        &mut encoder,
        pipeline,
        bind_group,
        prepared.groups,
        label,
    );
    checked_submit(device, label, encoder)
}
