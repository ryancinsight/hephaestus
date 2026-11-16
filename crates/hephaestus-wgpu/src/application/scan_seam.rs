use std::any::TypeId;
use std::marker::PhantomData;

use eunomia::Pod;
use hephaestus_core::{
    AxisScanMeta, BlockWidth, CombineExpr, DialectScalar, HephaestusError, IdentityToken,
    OpIdentity, Result, ScanDirection, ScanOps, StridedView, Wgsl, plan_axis_scan,
};

use crate::application::canary;
use crate::application::pipeline::try_cached_pipeline;
use crate::application::prepared::{
    checked_bind_group, checked_submit, device_owner, validate_device_owner,
};
use crate::application::scan::scan_shader_source;
use crate::application::strided::validate_out;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::{PipelineCache, WgpuDevice};

/// Provider-owned implementation of [`ScanOps`] over wgpu.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuScanOps;

/// Prepared resources for one scan dispatch bound to fixed operands.
pub struct PreparedScan {
    owner: PipelineCache,
    pipeline: Option<wgpu::ComputePipeline>,
    bind_group: Option<wgpu::BindGroup>,
    groups: u32,
    _meta_buffer: Option<crate::infrastructure::pool::PooledBuffer>,
}

impl<T> ScanOps<WgpuDevice, T> for WgpuScanOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    type Dialect = Wgsl;
    type PreparedScan<'op, const N: usize>
        = PreparedScan
    where
        T: 'op;

    fn prepare_scan_axis<'op, Op, const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<T>, N>,
        axis: usize,
        direction: ScanDirection,
        output: StridedView<'op, WgpuBuffer<T>, N>,
    ) -> Result<Self::PreparedScan<'op, N>>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>,
    {
        if N != 2 {
            return Err(HephaestusError::DispatchFailed {
                message: format!("scan currently supports only rank-2 operands, got rank {N}"),
            });
        }

        // SAFETY: N == 2 verified above; Layout<N> at runtime is Layout<2>.
        let input_layout: &leto::Layout<2> =
            unsafe { &*(input.layout as *const leto::Layout<N> as *const leto::Layout<2>) };
        let output_layout: &leto::Layout<2> =
            unsafe { &*(output.layout as *const leto::Layout<N> as *const leto::Layout<2>) };
        validate_out(output.buffer, output.layout)?;
        canary::require_template_safe::<T>(
            device,
            <Op as CombineExpr<Wgsl>>::EXPR,
            "prepared scan",
        )?;
        let Some(dispatch) = plan_axis_scan(
            input_layout,
            input.buffer.len,
            output_layout,
            output.buffer.len,
            axis,
            direction,
            BlockWidth::DEFAULT,
            input.buffer.aliases(output.buffer),
        )?
        else {
            return Ok(PreparedScan {
                owner: device_owner(device),
                pipeline: None,
                bind_group: None,
                groups: 0,
                _meta_buffer: None,
            });
        };

        let pipeline = try_cached_pipeline(
            device,
            (
                TypeId::of::<ScanSeamKernel<Op>>(),
                TypeId::of::<T>(),
                BlockWidth::DEFAULT.get(),
            ),
            "hephaestus-axis-scan-seam",
            || scan_shader_source::<Op, T>(BlockWidth::DEFAULT),
        )?;

        let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<AxisScanMeta>(1)?)?;

        device
            .queue()
            .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&dispatch.meta));

        let bind_group = checked_bind_group(
            device,
            &pipeline,
            "hephaestus-axis-scan-seam",
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer.as_entire_binding(),
                },
            ],
        )?;

        Ok(PreparedScan {
            owner: device_owner(device),
            pipeline: Some(pipeline),
            bind_group: Some(bind_group),
            groups: dispatch.groups,
            _meta_buffer: Some(meta_buffer),
        })
    }

    fn dispatch_scan<const N: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedScan<'_, N>,
    ) -> Result<()> {
        validate_device_owner(&prepared.owner, device, "scan")?;
        let Some(pipeline) = &prepared.pipeline else {
            return Ok(());
        };
        let Some(bind_group) = &prepared.bind_group else {
            return Ok(());
        };

        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-prepared-axis-scan"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hephaestus-prepared-axis-scan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(prepared.groups, 1, 1);
        }
        checked_submit(device, "hephaestus-prepared-axis-scan", encoder)
    }
}

struct ScanSeamKernel<Op>(PhantomData<Op>);
