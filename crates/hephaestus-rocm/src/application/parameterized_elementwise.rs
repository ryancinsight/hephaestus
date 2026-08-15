//! Device-neutral runtime-parameter unary seam for ROCm.

use hephaestus_core::{
    BlockWidth, DeviceBuffer, HephaestusError, HipC, ParameterizedUnaryExpr, ParameterizedUnaryOps,
    Result, StridedView, validate_parameterized_output,
};

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::application::strided::StridedOperand;
use crate::application::strided_elementwise::{
    HIP_DECODE, HIP_META, MAX_STRIDED_RANK, StridedMeta, dispatch_len, map_layout_err, pad_shape,
    pad_strides,
};
use crate::infrastructure::DevicePtr;
use crate::{RocmBuffer, RocmDevice};

fn parameterized_unary_shader<Op>() -> String
where
    Op: ParameterizedUnaryExpr<HipC>,
{
    format!(
        r#"
{meta}
extern "C" __global__ void parameterized_unary_strided_kernel(
    Meta lmeta,
    const {ty}* input,
    {ty} first,
    {ty} second,
    {ty}* output
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lmeta.offsets[3]) {{
        return;
    }}
{decode}
    {ty} x = input[a_offset];
    output[out_offset] = {expr};
}}
"#,
        meta = HIP_META,
        decode = HIP_DECODE,
        ty = "float",
        expr = Op::EXPR,
    )
}

fn launch_parameterized_unary<Op>(
    device: &RocmDevice,
    input: &RocmBuffer<f32>,
    parameters: [f32; 2],
    output: &RocmBuffer<f32>,
    meta: StridedMeta,
    width: BlockWidth,
    len: usize,
) -> Result<()>
where
    Op: ParameterizedUnaryExpr<HipC>,
{
    let key = PipelineKey::ParameterizedStridedUnary {
        op: core::any::TypeId::of::<Op>(),
        scalar: core::any::TypeId::of::<f32>(),
        width: width.get(),
    };
    let kernel = cached_kernel(
        device,
        key,
        "parameterized_unary_strided_kernel",
        parameterized_unary_shader::<Op>,
    )?;
    let mut meta = meta;
    let mut input_ptr: DevicePtr = input.raw();
    let [mut first, mut second] = parameters;
    let mut output_ptr: DevicePtr = output.raw();
    let mut args: [*mut core::ffi::c_void; 5] = [
        (&mut meta as *mut StridedMeta).cast(),
        (&mut input_ptr as *mut DevicePtr).cast(),
        (&mut first as *mut f32).cast(),
        (&mut second as *mut f32).cast(),
        (&mut output_ptr as *mut DevicePtr).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid_size(len, width)?, width),
        &mut args,
    )
}

/// Run a runtime-parameter unary expression over strided rank-`N` views.
pub fn parameterized_unary_strided_into<Op, const N: usize>(
    device: &RocmDevice,
    input: StridedOperand<'_, f32, N>,
    parameters: [f32; 2],
    output: StridedOperand<'_, f32, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: ParameterizedUnaryExpr<HipC>,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }
    let input_layout = input
        .layout
        .broadcast(output.layout.shape())
        .map_err(map_layout_err)?;
    input_layout
        .validate_storage_len(input.buffer.len())
        .map_err(map_layout_err)?;
    if input.buffer.aliases(output.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias input buffer".to_string(),
        });
    }
    let len = validate_parameterized_output(output.layout, output.buffer.len())?;
    if len == 0 {
        return Ok(());
    }
    let meta = StridedMeta {
        shape: pad_shape(output.layout.shape())?,
        a_strides: pad_strides(input_layout.strides())?,
        b_strides: [0; 4],
        out_strides: pad_strides(output.layout.strides())?,
        offsets: [
            u32::try_from(input_layout.offset()).map_err(|_| HephaestusError::DispatchFailed {
                message: "input offset exceeds u32 range".to_string(),
            })?,
            0,
            u32::try_from(output.layout.offset()).map_err(|_| HephaestusError::DispatchFailed {
                message: "output offset exceeds u32 range".to_string(),
            })?,
            dispatch_len(len)?,
        ],
    };
    launch_parameterized_unary::<Op>(
        device,
        input.buffer,
        parameters,
        output.buffer,
        meta,
        width,
        len,
    )
}

/// Provider-owned implementation of [`ParameterizedUnaryOps`] for ROCm.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmParameterizedUnaryOps;

impl ParameterizedUnaryOps<RocmDevice> for RocmParameterizedUnaryOps {
    type Dialect = HipC;

    fn parameterized_unary_into<Op, const N: usize>(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<f32>, N>,
        parameters: [f32; 2],
        output: StridedView<'_, RocmBuffer<f32>, N>,
    ) -> Result<()>
    where
        Op: ParameterizedUnaryExpr<Self::Dialect>,
    {
        parameterized_unary_strided_into::<Op, N>(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
            parameters,
            StridedOperand {
                buffer: output.buffer,
                layout: output.layout,
            },
            BlockWidth::DEFAULT,
        )
    }
}

pub use hephaestus_core::{HardtanhGradOp, HardtanhOp, ThresholdGradOp, ThresholdOp};
