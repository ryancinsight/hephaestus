//! Runtime-rank expression fusion implemented by the Metal provider.
//!
//! Metal executes through its inner WGPU device, so fusion delegates to
//! [`WgpuFusionOps`](hephaestus_wgpu::WgpuFusionOps) after unwrapping the
//! Metal views. The dialect is WGSL on both sides; no second shader
//! language is involved.

use hephaestus_core::{
    DynamicStridedView, FusedElementwiseOps, FusedExpression, FusedReduction, FusedReductionOps,
    FusionScalar, Result, Wgsl,
};
use hephaestus_wgpu::{WgpuBuffer, WgpuFusionOps};

use crate::{MetalBuffer, MetalDevice};

fn inner_views<'b, T>(
    views: &[DynamicStridedView<'b, MetalBuffer<T>>],
) -> Vec<DynamicStridedView<'b, WgpuBuffer<T>>> {
    views
        .iter()
        .map(|view| DynamicStridedView {
            buffer: view.buffer.wgpu_buffer(),
            layout: view.layout,
        })
        .collect()
}

fn inner_output<T>(
    output: DynamicStridedView<'_, MetalBuffer<T>>,
) -> DynamicStridedView<'_, WgpuBuffer<T>> {
    DynamicStridedView {
        buffer: output.buffer.wgpu_buffer(),
        layout: output.layout,
    }
}

/// Static provider value implementing both runtime fusion roles.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalFusionOps;

impl<T> FusedElementwiseOps<MetalDevice, T> for MetalFusionOps
where
    T: FusionScalar<Wgsl>,
{
    type Dialect = Wgsl;

    fn fused_elementwise_into<E>(
        &self,
        device: &MetalDevice,
        expression: &E,
        inputs: &[DynamicStridedView<'_, MetalBuffer<T>>],
        output: DynamicStridedView<'_, MetalBuffer<T>>,
    ) -> Result<()>
    where
        E: FusedExpression<Self::Dialect>,
    {
        WgpuFusionOps.fused_elementwise_into(
            &device.inner,
            expression,
            &inner_views(inputs),
            inner_output(output),
        )
    }
}

impl<T> FusedReductionOps<MetalDevice, T> for MetalFusionOps
where
    T: FusionScalar<Wgsl>,
{
    type Dialect = Wgsl;

    fn fused_reduce_into<E>(
        &self,
        device: &MetalDevice,
        expression: &E,
        inputs: &[DynamicStridedView<'_, MetalBuffer<T>>],
        reduction: FusedReduction,
        axis: usize,
        output: DynamicStridedView<'_, MetalBuffer<T>>,
    ) -> Result<()>
    where
        E: FusedExpression<Self::Dialect>,
    {
        WgpuFusionOps.fused_reduce_into(
            &device.inner,
            expression,
            &inner_views(inputs),
            reduction,
            axis,
            inner_output(output),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use hephaestus_core::{ComputeDevice, FusedExpression, Wgsl};

    use super::*;

    struct AddExpression;

    impl FusedExpression<Wgsl> for AddExpression {
        fn source(&self) -> Cow<'static, str> {
            Cow::Borrowed("input_0 + input_1")
        }
    }

    /// The Metal delegation runs the inner WGPU fusion verbatim.
    ///
    /// This wraps a real host-platform WGPU device rather than requiring
    /// macOS Metal, so the view mapping is covered on every platform; the
    /// macOS-only `fusion_contracts` test repeats the full conformance
    /// suite on a physical Metal device.
    #[test]
    fn delegation_evaluates_the_inner_fusion() {
        let inner = match hephaestus_wgpu::WgpuDevice::try_default("hephaestus-metal-fusion-test") {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skip Metal fusion delegation test: {error}");
                return;
            }
        };
        let device = MetalDevice {
            inner: inner.clone(),
        };
        let first = MetalBuffer {
            inner: inner.upload(&[1.0f32, 2.0]).expect("first input"),
        };
        let second = MetalBuffer {
            inner: inner.upload(&[10.0f32, 20.0]).expect("second input"),
        };
        let output = MetalBuffer {
            inner: inner.alloc_zeroed::<f32>(2).expect("output"),
        };
        let layout = leto::LayoutDyn::new(
            vec![2usize].into_boxed_slice(),
            vec![1isize].into_boxed_slice(),
            0,
        )
        .expect("test layout");
        MetalFusionOps
            .fused_elementwise_into(
                &device,
                &AddExpression,
                &[
                    DynamicStridedView::new(&first, &layout),
                    DynamicStridedView::new(&second, &layout),
                ],
                DynamicStridedView::new(&output, &layout),
            )
            .expect("delegated fusion dispatch");
        let mut values = vec![0.0f32; 2];
        inner
            .download(output.wgpu_buffer(), &mut values)
            .expect("download");
        assert_eq!(values, [11.0, 22.0]);
    }
}
