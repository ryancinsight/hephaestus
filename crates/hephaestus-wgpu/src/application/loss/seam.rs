use core::any::TypeId;

use hephaestus_core::{
    ComputeDevice, CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyOps,
    DeviceFeature, HephaestusError, Result, plan_cross_entropy_backward,
    plan_cross_entropy_forward,
};

use super::metadata::CrossEntropyMeta;
use super::prepared::{
    PreparedCrossEntropyBackward, PreparedCrossEntropyForward, PreparedCrossEntropyKernel,
};
use super::resources::{
    address_limit, backward_aliases, binding, forward_aliases, metadata_buffer, raw_binding,
    validate_backward_owners, validate_forward_owners,
};
use super::shader::{
    BACKWARD_ACCUMULATE, BACKWARD_ARITHMETIC, BACKWARD_ROWS, FORWARD_MEAN, FORWARD_PREFLIGHT,
    FORWARD_ROWS, WgslCrossEntropyScalar, shader,
};
use crate::application::canary::{self, CanaryWidth, MathBuiltin};
use crate::application::pipeline::{try_cached_pipeline, workgroups};
use crate::application::prepared::checked_bind_group;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

const WORKGROUP_WIDTH: hephaestus_core::BlockWidth = hephaestus_core::BlockWidth::DEFAULT;

struct ForwardPreflight;
struct ForwardRows;
struct ForwardMean;
struct BackwardRows;
struct BackwardArithmetic;
struct BackwardAccumulate;

/// WGPU implementation of provider-owned mean cross-entropy.
///
/// Runtime adapter selection makes this implementation serve native Vulkan,
/// DirectX, browser WebGPU, and Metal devices without a parallel Metal path.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuCrossEntropyOps;

fn prepare_forward<'a, T: WgslCrossEntropyScalar>(
    device: &'a WgpuDevice,
    operands: CrossEntropyForwardOperands<'a, WgpuBuffer<T>, WgpuBuffer<u32>>,
) -> Result<PreparedCrossEntropyForward<T>> {
    validate_forward_owners(device, &operands)?;
    let plan = plan_cross_entropy_forward(&operands, forward_aliases(&operands))?;
    plan.validate_address_limit(address_limit())?;
    let metadata = CrossEntropyMeta::forward::<T>(
        plan,
        operands.logits.layout,
        operands.targets.layout,
        operands.loss.layout,
        operands.probabilities.layout,
    )?;
    let status = device.alloc_zeroed::<u32>(1)?;
    let row_losses = device.alloc_zeroed::<T>(plan.batch)?;
    let preflight = prepare::<ForwardPreflight, T>(
        device,
        &metadata,
        plan.batch,
        FORWARD_PREFLIGHT,
        "hephaestus-cross-entropy-forward-preflight",
        &[
            binding(0, operands.logits.buffer),
            binding(1, operands.targets.buffer),
            binding(2, &status),
        ],
    )?;
    let probabilities = prepare::<ForwardRows, T>(
        device,
        &metadata,
        plan.batch,
        FORWARD_ROWS,
        "hephaestus-cross-entropy-forward-rows",
        &[
            binding(0, operands.logits.buffer),
            binding(1, operands.targets.buffer),
            binding(2, operands.probabilities.buffer),
            binding(3, &row_losses),
        ],
    )?;
    let mean = prepare::<ForwardMean, T>(
        device,
        &metadata,
        1,
        FORWARD_MEAN,
        "hephaestus-cross-entropy-forward-mean",
        &[binding(0, &row_losses), binding(1, operands.loss.buffer)],
    )?;
    Ok(PreparedCrossEntropyForward::new(
        preflight,
        status,
        probabilities,
        mean,
        row_losses,
    ))
}

fn prepare_backward<'a, T: WgslCrossEntropyScalar>(
    device: &'a WgpuDevice,
    operands: CrossEntropyBackwardOperands<'a, WgpuBuffer<T>, WgpuBuffer<u32>>,
) -> Result<PreparedCrossEntropyBackward> {
    validate_backward_owners(device, &operands)?;
    // The arithmetic preflight binds four storage buffers (probabilities,
    // targets, destination, status) plus the upstream re-read binding when
    // the width declares one; reject under-provisioned devices with a
    // configuration error instead of a pipeline-layout validation failure.
    require_arithmetic_storage(device, 4 + u32::from(T::ARITHMETIC_UPSTREAM.is_some()))?;
    let plan = plan_cross_entropy_backward(&operands, backward_aliases(&operands))?;
    plan.validate_address_limit(address_limit())?;
    let metadata = CrossEntropyMeta::backward::<T>(
        plan,
        operands.output_gradient.layout,
        operands.probabilities.layout,
        operands.targets.layout,
        operands.logit_gradient.layout,
    )?;
    let status = device.alloc_zeroed::<u32>(T::BACKWARD_STATUS_WORDS)?;
    let row_preflight = prepare::<BackwardRows, T>(
        device,
        &metadata,
        plan.batch,
        BACKWARD_ROWS,
        "hephaestus-cross-entropy-backward-row-preflight",
        &[
            binding(0, operands.output_gradient.buffer),
            binding(1, operands.probabilities.buffer),
            binding(2, operands.targets.buffer),
            binding(3, &status),
        ],
    )?;
    // f64 re-reads the upstream gradient from its buffer (the status words
    // cannot round-trip f64 bits: naga rejects f64 bitcasts), so the
    // arithmetic preflight takes the extra storage binding the shader
    // declares; the uniform binding follows positionally. The entries die at
    // the block end so the later `status` move is borrow-free.
    let arithmetic_preflight = {
        let mut arithmetic_entries: smallvec::SmallVec<[wgpu::BindGroupEntry<'_>; 5]> = [
            binding(0, operands.probabilities.buffer),
            binding(1, operands.targets.buffer),
            binding(2, operands.logit_gradient.buffer),
            binding(3, &status),
        ]
        .into_iter()
        .collect();
        if let Some(upstream) = T::ARITHMETIC_UPSTREAM {
            arithmetic_entries.push(binding(upstream, operands.output_gradient.buffer));
        }
        prepare::<BackwardArithmetic, T>(
            device,
            &metadata,
            plan.elements,
            BACKWARD_ARITHMETIC,
            "hephaestus-cross-entropy-backward-arithmetic-preflight",
            &arithmetic_entries,
        )?
    };
    let backward = prepare::<BackwardAccumulate, T>(
        device,
        &metadata,
        plan.elements,
        BACKWARD_ACCUMULATE,
        "hephaestus-cross-entropy-backward-accumulate",
        &[
            binding(0, operands.output_gradient.buffer),
            binding(1, operands.probabilities.buffer),
            binding(2, operands.targets.buffer),
            binding(3, operands.logit_gradient.buffer),
        ],
    )?;
    Ok(PreparedCrossEntropyBackward::new(
        row_preflight,
        arithmetic_preflight,
        status,
        backward,
    ))
}

impl CrossEntropyOps<WgpuDevice, f32> for WgpuCrossEntropyOps {
    type PreparedForward<'a>
        = PreparedCrossEntropyForward<f32>
    where
        WgpuDevice: 'a,
        f32: 'a;
    type PreparedBackward<'a>
        = PreparedCrossEntropyBackward
    where
        WgpuDevice: 'a,
        f32: 'a;

    fn prepare_cross_entropy_forward<'a>(
        &self,
        device: &'a WgpuDevice,
        operands: CrossEntropyForwardOperands<'a, WgpuBuffer<f32>, WgpuBuffer<u32>>,
    ) -> Result<Self::PreparedForward<'a>> {
        prepare_forward(device, operands)
    }

    fn dispatch_cross_entropy_forward(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_cross_entropy_backward<'a>(
        &self,
        device: &'a WgpuDevice,
        operands: CrossEntropyBackwardOperands<'a, WgpuBuffer<f32>, WgpuBuffer<u32>>,
    ) -> Result<Self::PreparedBackward<'a>> {
        prepare_backward(device, operands)
    }

    fn dispatch_cross_entropy_backward(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}

impl CrossEntropyOps<WgpuDevice, f64> for WgpuCrossEntropyOps {
    type PreparedForward<'a>
        = PreparedCrossEntropyForward<f64>
    where
        WgpuDevice: 'a,
        f64: 'a;
    type PreparedBackward<'a>
        = PreparedCrossEntropyBackward
    where
        WgpuDevice: 'a,
        f64: 'a;

    fn prepare_cross_entropy_forward<'a>(
        &self,
        device: &'a WgpuDevice,
        operands: CrossEntropyForwardOperands<'a, WgpuBuffer<f64>, WgpuBuffer<u32>>,
    ) -> Result<Self::PreparedForward<'a>> {
        require_shader_f64(device)?;
        // The forward shaders call f64 `log`, which aborts pipeline
        // compilation on the recorded adapters; refuse with cause instead.
        canary::require_builtin(
            device,
            MathBuiltin::Log,
            CanaryWidth::F64,
            "cross-entropy forward",
        )?;
        prepare_forward(device, operands)
    }

    fn dispatch_cross_entropy_forward(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_cross_entropy_backward<'a>(
        &self,
        device: &'a WgpuDevice,
        operands: CrossEntropyBackwardOperands<'a, WgpuBuffer<f64>, WgpuBuffer<u32>>,
    ) -> Result<Self::PreparedBackward<'a>> {
        require_shader_f64(device)?;
        prepare_backward(device, operands)
    }

    fn dispatch_cross_entropy_backward(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}

fn require_shader_f64(device: &WgpuDevice) -> Result<()> {
    if device.supports_device_feature(DeviceFeature::ShaderF64) {
        Ok(())
    } else {
        Err(HephaestusError::InvalidConfiguration {
            message: "WGPU cross-entropy requires the ShaderF64 device feature for f64".to_string(),
        })
    }
}

fn require_arithmetic_storage(device: &WgpuDevice, required_storage: u32) -> Result<()> {
    let limits = device.limits();
    // WGPU 30 counts the uniform buffer against the combined budget too.
    let required_combined = required_storage + 1;
    if limits.max_storage_buffers_per_shader_stage >= required_storage
        && limits.max_buffers_and_acceleration_structures_per_shader_stage >= required_combined
    {
        Ok(())
    } else {
        Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "WGPU cross-entropy backward requires {required_storage} storage buffers \
                 ({required_combined} combined with the uniform buffer) per shader stage"
            ),
        })
    }
}

fn prepare<K: 'static, T: WgslCrossEntropyScalar>(
    device: &WgpuDevice,
    metadata: &CrossEntropyMeta,
    elements: usize,
    stage: u8,
    label: &'static str,
    storage_entries: &[wgpu::BindGroupEntry<'_>],
) -> Result<PreparedCrossEntropyKernel> {
    let pipeline = try_cached_pipeline(
        device,
        (TypeId::of::<K>(), TypeId::of::<T>(), WORKGROUP_WIDTH.get()),
        label,
        || shader::<T>(stage, WORKGROUP_WIDTH.get()),
    )?;
    let metadata_buffer = metadata_buffer(device, metadata)?;
    let mut entries: smallvec::SmallVec<[wgpu::BindGroupEntry<'_>; 6]> =
        storage_entries.iter().cloned().collect();
    entries.push(raw_binding(
        u32::try_from(storage_entries.len())
            .expect("invariant: cross-entropy binding count fits u32"),
        &metadata_buffer,
    ));
    let bind_group = checked_bind_group(device, &pipeline, label, &entries)?;
    drop(entries);
    Ok(PreparedCrossEntropyKernel::new(
        device,
        pipeline,
        bind_group,
        metadata_buffer,
        workgroups(elements, WORKGROUP_WIDTH)?,
        label,
    ))
}
