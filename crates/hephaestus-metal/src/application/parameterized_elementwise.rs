//! Runtime-parameter unary dispatch through WGPU's native Metal path.

use hephaestus_core::{
    BlockWidth, ParameterizedUnaryExpr, ParameterizedUnaryOps, Result, StridedView, Wgsl,
};
use hephaestus_wgpu::{WgpuDevice, WgpuParameterizedUnaryOps};

use crate::application::strided::{StridedOperand, to_wgpu_strided};
use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`ParameterizedUnaryOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalParameterizedUnaryOps;

/// Run a runtime-parameter unary expression over Metal-backed strided views.
///
/// # Errors
///
/// Returns a layout, shape, alias, pipeline, or dispatch error from the native
/// Metal-selected WGPU provider.
pub fn parameterized_unary_strided_into<Op, const N: usize>(
    device: &MetalDevice,
    input: StridedOperand<'_, f32, N>,
    parameters: [f32; 2],
    output: StridedOperand<'_, f32, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: ParameterizedUnaryExpr<Wgsl>,
{
    hephaestus_wgpu::parameterized_unary_strided_into::<Op, N>(
        device.wgpu_device(),
        to_wgpu_strided(input),
        parameters,
        to_wgpu_strided(output),
        width,
    )
}

impl ParameterizedUnaryOps<MetalDevice> for MetalParameterizedUnaryOps {
    type Dialect = Wgsl;

    fn parameterized_unary_into<Op, const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<f32>, N>,
        parameters: [f32; 2],
        output: StridedView<'_, MetalBuffer<f32>, N>,
    ) -> Result<()>
    where
        Op: ParameterizedUnaryExpr<Self::Dialect>,
    {
        <WgpuParameterizedUnaryOps as ParameterizedUnaryOps<WgpuDevice>>::parameterized_unary_into::<
            Op,
            N,
        >(
            &WgpuParameterizedUnaryOps,
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            parameters,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
