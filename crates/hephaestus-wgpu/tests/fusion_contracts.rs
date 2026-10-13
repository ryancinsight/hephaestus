//! Value-semantic contracts for provider-owned runtime-rank fusion.

use std::borrow::Cow;

use super::device_features::device_with_optional_features;
use hephaestus_conformance::assert_fusion_contract;
use hephaestus_core::{
    ComputeDevice, DeviceBuffer, DeviceFeature, DynamicStridedView, FusedElementwiseOps,
    FusedExpression, FusedReduction, FusedReductionOps, HephaestusError, Wgsl,
};
use hephaestus_wgpu::{WgpuDevice, WgpuFusionOps};
use leto::LayoutDyn;

struct AddExpression;

impl FusedExpression<Wgsl> for AddExpression {
    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed("input_0 + input_1")
    }
}

struct InputExpression;

impl FusedExpression<Wgsl> for InputExpression {
    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed("input_0")
    }
}

fn device_or_skip() -> Option<WgpuDevice> {
    super::device_or_skip()
}

fn layout(shape: &[usize], strides: &[isize]) -> LayoutDyn {
    LayoutDyn::new(
        shape.to_vec().into_boxed_slice(),
        strides.to_vec().into_boxed_slice(),
        0,
    )
    .expect("fusion test layout ranks match")
}

fn download(device: &WgpuDevice, buffer: &hephaestus_wgpu::WgpuBuffer<f32>) -> Vec<f32> {
    let mut values = vec![0.0; buffer.len()];
    device
        .download(buffer, &mut values)
        .expect("fusion download");
    values
}

pub(super) fn runtime_rank_elementwise_broadcast_matches_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let first = device.upload(&[1.0_f32, 2.0]).expect("first input");
    let second = device
        .upload(&[10.0_f32, 20.0, 30.0])
        .expect("second input");
    let output = device.alloc_zeroed::<f32>(6).expect("output allocation");
    let first_layout = layout(&[2, 1], &[1, 1]);
    let second_layout = layout(&[1, 3], &[3, 1]);
    let output_layout = layout(&[2, 3], &[3, 1]);

    WgpuFusionOps
        .fused_elementwise_into(
            &device,
            &AddExpression,
            &[
                DynamicStridedView::new(&first, &first_layout),
                DynamicStridedView::new(&second, &second_layout),
            ],
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect("runtime-rank elementwise dispatch");

    assert_eq!(
        download(&device, &output),
        [11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
    );
}

pub(super) fn runtime_rank_reduction_matches_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let input = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("reduction input");
    let output = device.alloc_zeroed::<f32>(2).expect("reduction output");
    let input_layout = layout(&[2, 3], &[3, 1]);
    let output_layout = layout(&[2, 1], &[1, 1]);

    WgpuFusionOps
        .fused_reduce_into(
            &device,
            &InputExpression,
            &[DynamicStridedView::new(&input, &input_layout)],
            FusedReduction::Sum,
            1,
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect("runtime-rank reduction dispatch");

    assert_eq!(download(&device, &output), [6.0, 15.0]);
}

pub(super) fn runtime_rank_empty_sum_uses_identity_and_mean_rejects_empty_axis() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let input = device
        .alloc_uninitialized::<f32>(0)
        .expect("empty reduction input");
    let output = device.upload(&[7.0_f32, 11.0]).expect("reduction output");
    let input_layout = layout(&[2, 0], &[0, 1]);
    let output_layout = layout(&[2, 1], &[1, 1]);

    WgpuFusionOps
        .fused_reduce_into(
            &device,
            &InputExpression,
            &[DynamicStridedView::new(&input, &input_layout)],
            FusedReduction::Sum,
            1,
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect("empty sum identity dispatch");
    assert_eq!(download(&device, &output), [0.0, 0.0]);

    let error = WgpuFusionOps
        .fused_reduce_into(
            &device,
            &InputExpression,
            &[DynamicStridedView::new(&input, &input_layout)],
            FusedReduction::Mean,
            1,
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect_err("empty mean must be rejected");
    assert!(
        matches!(error, HephaestusError::InvalidConfiguration { .. }),
        "unexpected error: {error}"
    );
    assert_eq!(download(&device, &output), [0.0, 0.0]);
}

pub(super) fn fusion_rejects_noninjective_output() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let input = device.upload(&[1.0_f32, 2.0]).expect("input");
    let output = device.upload(&[7.0_f32]).expect("output");
    let input_layout = layout(&[2], &[1]);
    let output_layout = layout(&[2], &[0]);

    let error = WgpuFusionOps
        .fused_elementwise_into(
            &device,
            &InputExpression,
            &[DynamicStridedView::new(&input, &input_layout)],
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect_err("non-injective fusion output must be rejected");

    assert!(
        matches!(error, HephaestusError::DispatchFailed { .. }),
        "unexpected error: {error}"
    );
    assert_eq!(download(&device, &output), [7.0]);
}

pub(super) fn wgpu_satisfies_the_fusion_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_fusion_contract::<_, _, _, f32, _, _>(
        &device,
        &WgpuFusionOps,
        &AddExpression,
        &InputExpression,
    );
}

pub(super) fn wgpu_fusion_respects_precision_capabilities() {
    let Some(device) = device_with_optional_features(
        "hephaestus-fusion-precision-contract",
        &[DeviceFeature::ShaderF64],
    ) else {
        return;
    };
    if !device.supports_device_feature(DeviceFeature::ShaderF64) {
        let original = [2.0_f64, 3.0, 5.0, 7.0];
        let input = device.upload(&original).expect("input");
        let output = device.upload(&[11.0_f64; 4]).expect("output");
        let input_layout = layout(&[4], &[1]);
        let output_layout = layout(&[4], &[1]);
        let scalar_output = device.upload(&[13.0_f64]).expect("scalar output");
        let scalar_layout = layout(&[1], &[1]);
        let inputs = [DynamicStridedView::new(&input, &input_layout)];
        let failures = [
            WgpuFusionOps.fused_elementwise_into(
                &device,
                &InputExpression,
                &inputs,
                DynamicStridedView::new(&output, &output_layout),
            ),
            WgpuFusionOps.fused_reduce_into(
                &device,
                &InputExpression,
                &inputs,
                FusedReduction::Sum,
                0,
                DynamicStridedView::new(&scalar_output, &scalar_layout),
            ),
        ];
        for failure in failures {
            match failure {
                Err(HephaestusError::DispatchFailed { message }) => {
                    assert!(message.contains("FLOAT64"), "{message}");
                }
                other => panic!("unsupported native precision must reject fusion: {other:?}"),
            }
        }
        assert_eq!(
            device.download_owned(&input).expect("input readback"),
            original
        );
        assert_eq!(
            device.download_owned(&output).expect("output readback"),
            [11.0; 4]
        );
        assert_eq!(
            device
                .download_owned(&scalar_output)
                .expect("scalar readback"),
            [13.0]
        );
        return;
    }
    assert_fusion_contract::<_, _, _, f64, _, _>(
        &device,
        &WgpuFusionOps,
        &AddExpression,
        &InputExpression,
    );
}
