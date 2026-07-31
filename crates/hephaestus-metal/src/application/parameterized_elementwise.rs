//! Runtime-parameter unary dispatch through WGPU's native Metal path.

use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, DialectScalar, ParameterizedUnaryExpr, ParameterizedUnaryOps, Result, StridedView,
    Wgsl,
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
pub fn parameterized_unary_strided_into<Op, T, const N: usize>(
    device: &MetalDevice,
    input: StridedOperand<'_, T, N>,
    parameters: [T; 2],
    output: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: ParameterizedUnaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod,
{
    hephaestus_wgpu::parameterized_unary_strided_into::<Op, T, N>(
        device.wgpu_device(),
        to_wgpu_strided(input),
        parameters,
        to_wgpu_strided(output),
        width,
    )
}

impl<T> ParameterizedUnaryOps<MetalDevice, T> for MetalParameterizedUnaryOps
where
    T: DialectScalar<Wgsl> + Pod + Send + Sync + 'static,
{
    type Dialect = Wgsl;

    fn parameterized_unary_into<Op, const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        parameters: [T; 2],
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<()>
    where
        Op: ParameterizedUnaryExpr<Self::Dialect>,
    {
        <WgpuParameterizedUnaryOps as ParameterizedUnaryOps<WgpuDevice, T>>::parameterized_unary_into::<
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
